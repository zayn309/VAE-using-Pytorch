# Conditional VAE for Facial Expressions (PyTorch)

A **Conditional Variational Autoencoder (CVAE)** built from scratch in PyTorch that learns a latent space of human faces and can reconstruct and generate faces conditioned on a facial-expression class.

All the work lives in [`VAE.ipynb`](VAE.ipynb), designed to run on Google Colab with a GPU.

## Pipeline

1. **Face extraction** — faces are detected and cropped from raw photos with **MTCNN** (`facenet-pytorch`), producing a clean, aligned face dataset organised into one folder per expression.
2. **Dataset** — a custom `ExpressionsDataset` loads the cropped faces as 256×256 RGB tensors together with their expression label.
3. **Model** — a convolutional **encoder** maps an image and its one-hot class label to the mean and log-variance of a 400-dimensional latent distribution. The **decoder** maps a sampled latent vector plus the class label back to an image.
4. **Loss** — reconstruction loss plus **KL divergence** (custom `CvaeLoss`), using the reparameterisation trick so the model can be trained end-to-end.
5. **Training and visualisation** — Adam (lr = 1e-4), batch size 32, up to 100 epochs, then reconstructions of random samples are plotted next to the originals.

## Key concepts

- Variational inference and the reparameterisation trick
- Conditioning both encoder and decoder on a class label
- Balancing reconstruction quality against latent-space regularity (KL term)

## Getting started

```bash
pip install torch torchvision facenet-pytorch matplotlib numpy pandas tqdm
```

Open `VAE.ipynb` in Jupyter or Colab, point the dataset paths at your own folder of face images (one sub-folder per expression), and run the cells in order.

## License

[MIT](LICENSE)

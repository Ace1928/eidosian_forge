import joblib
import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KDTree, KernelDensity, NearestNeighbors
from sklearn.neighbors._ball_tree import kernel_norm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import assert_allclose
def compute_kernel_slow(Y, X, kernel, h):
    if h == 'scott':
        h = X.shape[0] ** (-1 / (X.shape[1] + 4))
    elif h == 'silverman':
        h = (X.shape[0] * (X.shape[1] + 2) / 4) ** (-1 / (X.shape[1] + 4))
    d = np.sqrt(((Y[:, None, :] - X) ** 2).sum(-1))
    norm = kernel_norm(h, X.shape[1], kernel) / X.shape[0]
    if kernel == 'gaussian':
        return norm * np.exp(-0.5 * (d * d) / (h * h)).sum(-1)
    elif kernel == 'tophat':
        return norm * (d < h).sum(-1)
    elif kernel == 'epanechnikov':
        return norm * ((1.0 - d * d / (h * h)) * (d < h)).sum(-1)
    elif kernel == 'exponential':
        return norm * np.exp(-d / h).sum(-1)
    elif kernel == 'linear':
        return norm * ((1 - d / h) * (d < h)).sum(-1)
    elif kernel == 'cosine':
        return norm * (np.cos(0.5 * np.pi * d / h) * (d < h)).sum(-1)
    else:
        raise ValueError('kernel not recognized')
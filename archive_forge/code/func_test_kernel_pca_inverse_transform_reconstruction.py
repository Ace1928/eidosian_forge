import warnings
import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_iris, make_blobs, make_circles
from sklearn.decomposition import PCA, KernelPCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Perceptron
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import _check_psd_eigenvalues
def test_kernel_pca_inverse_transform_reconstruction():
    """Test if the reconstruction is a good approximation.

    Note that in general it is not possible to get an arbitrarily good
    reconstruction because of kernel centering that does not
    preserve all the information of the original data.
    """
    X, *_ = make_blobs(n_samples=100, n_features=4, random_state=0)
    kpca = KernelPCA(n_components=20, kernel='rbf', fit_inverse_transform=True, alpha=0.001)
    X_trans = kpca.fit_transform(X)
    X_reconst = kpca.inverse_transform(X_trans)
    assert np.linalg.norm(X - X_reconst) / np.linalg.norm(X) < 0.1
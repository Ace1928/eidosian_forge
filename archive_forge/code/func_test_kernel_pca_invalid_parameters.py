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
def test_kernel_pca_invalid_parameters():
    """Check that kPCA raises an error if the parameters are invalid

    Tests fitting inverse transform with a precomputed kernel raises a
    ValueError.
    """
    estimator = KernelPCA(n_components=10, fit_inverse_transform=True, kernel='precomputed')
    err_ms = 'Cannot fit_inverse_transform with a precomputed kernel'
    with pytest.raises(ValueError, match=err_ms):
        estimator.fit(np.random.randn(10, 10))
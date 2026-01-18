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
def test_kernel_pca_feature_names_out():
    """Check feature names out for KernelPCA."""
    X, *_ = make_blobs(n_samples=100, n_features=4, random_state=0)
    kpca = KernelPCA(n_components=2).fit(X)
    names = kpca.get_feature_names_out()
    assert_array_equal([f'kernelpca{i}' for i in range(2)], names)
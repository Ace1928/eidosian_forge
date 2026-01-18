import re
import warnings
import numpy as np
import pytest
import scipy as sp
from numpy.testing import assert_array_equal
from sklearn import config_context, datasets
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.decomposition import PCA
from sklearn.decomposition._pca import _assess_dimension, _infer_dimension
from sklearn.utils._array_api import (
from sklearn.utils._array_api import device as array_device
from sklearn.utils._testing import _array_api_for_tests, assert_allclose
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_mle_simple_case():
    n_samples, n_dim = (1000, 10)
    X = np.random.RandomState(0).randn(n_samples, n_dim)
    X[:, -1] = np.mean(X[:, :-1], axis=-1)
    pca_skl = PCA('mle', svd_solver='full')
    pca_skl.fit(X)
    assert pca_skl.n_components_ == n_dim - 1
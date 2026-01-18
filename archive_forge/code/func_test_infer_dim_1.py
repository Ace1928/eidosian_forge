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
def test_infer_dim_1():
    n, p = (1000, 5)
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1 + rng.randn(n, 1) * np.array([3, 4, 5, 1, 2]) + np.array([1, 0, 7, 4, 6])
    pca = PCA(n_components=p, svd_solver='full')
    pca.fit(X)
    spect = pca.explained_variance_
    ll = np.array([_assess_dimension(spect, k, n) for k in range(1, p)])
    assert ll[1] > ll.max() - 0.01 * n
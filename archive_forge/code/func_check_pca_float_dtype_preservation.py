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
def check_pca_float_dtype_preservation(svd_solver):
    X_64 = np.random.RandomState(0).rand(1000, 4).astype(np.float64, copy=False)
    X_32 = X_64.astype(np.float32)
    pca_64 = PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(X_64)
    pca_32 = PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(X_32)
    assert pca_64.components_.dtype == np.float64
    assert pca_32.components_.dtype == np.float32
    assert pca_64.transform(X_64).dtype == np.float64
    assert pca_32.transform(X_32).dtype == np.float32
    assert_allclose(pca_64.components_, pca_32.components_, rtol=0.0002)
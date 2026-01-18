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
@pytest.mark.parametrize('X, n_components, n_components_validated', [(iris.data, 0.95, 2), (iris.data, 0.01, 1), (np.random.RandomState(0).rand(5, 20), 0.5, 2)])
def test_infer_dim_by_explained_variance(X, n_components, n_components_validated):
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(X)
    assert pca.n_components == pytest.approx(n_components)
    assert pca.n_components_ == n_components_validated
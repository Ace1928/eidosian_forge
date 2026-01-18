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
def test_array_api_error_and_warnings_on_unsupported_params():
    pytest.importorskip('array_api_compat')
    xp = pytest.importorskip('numpy.array_api')
    iris_xp = xp.asarray(iris.data)
    pca = PCA(n_components=2, svd_solver='arpack', random_state=0)
    expected_msg = re.escape("PCA with svd_solver='arpack' is not supported for Array API inputs.")
    with pytest.raises(ValueError, match=expected_msg):
        with config_context(array_api_dispatch=True):
            pca.fit(iris_xp)
    pca.set_params(svd_solver='randomized', power_iteration_normalizer='LU')
    expected_msg = re.escape("Array API does not support LU factorization. Set `power_iteration_normalizer='QR'` instead.")
    with pytest.raises(ValueError, match=expected_msg):
        with config_context(array_api_dispatch=True):
            pca.fit(iris_xp)
    pca.set_params(svd_solver='randomized', power_iteration_normalizer='auto')
    expected_msg = re.escape("Array API does not support LU factorization, falling back to QR instead. Set `power_iteration_normalizer='QR'` explicitly to silence this warning.")
    with pytest.warns(UserWarning, match=expected_msg):
        with config_context(array_api_dispatch=True):
            pca.fit(iris_xp)
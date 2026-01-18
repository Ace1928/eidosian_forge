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
def check_array_api_get_precision(name, estimator, array_namespace, device, dtype_name):
    xp = _array_api_for_tests(array_namespace, device)
    iris_np = iris.data.astype(dtype_name)
    iris_xp = xp.asarray(iris_np, device=device)
    estimator.fit(iris_np)
    precision_np = estimator.get_precision()
    covariance_np = estimator.get_covariance()
    with config_context(array_api_dispatch=True):
        estimator_xp = clone(estimator).fit(iris_xp)
        precision_xp = estimator_xp.get_precision()
        assert precision_xp.shape == (4, 4)
        assert precision_xp.dtype == iris_xp.dtype
        assert_allclose(_convert_to_numpy(precision_xp, xp=xp), precision_np, atol=_atol_for_type(dtype_name))
        covariance_xp = estimator_xp.get_covariance()
        assert covariance_xp.shape == (4, 4)
        assert covariance_xp.dtype == iris_xp.dtype
        assert_allclose(_convert_to_numpy(covariance_xp, xp=xp), covariance_np, atol=_atol_for_type(dtype_name))
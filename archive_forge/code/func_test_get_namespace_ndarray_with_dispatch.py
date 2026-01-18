from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
@skip_if_array_api_compat_not_configured
def test_get_namespace_ndarray_with_dispatch():
    """Test get_namespace on NumPy ndarrays."""
    array_api_compat = pytest.importorskip('array_api_compat')
    X_np = numpy.asarray([[1, 2, 3]])
    with config_context(array_api_dispatch=True):
        xp_out, is_array_api_compliant = get_namespace(X_np)
        assert is_array_api_compliant
        assert xp_out is array_api_compat.numpy
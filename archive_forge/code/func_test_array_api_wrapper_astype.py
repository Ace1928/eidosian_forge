from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
def test_array_api_wrapper_astype():
    """Test _ArrayAPIWrapper for ArrayAPIs that is not NumPy."""
    numpy_array_api = pytest.importorskip('numpy.array_api')
    xp_ = _AdjustableNameAPITestWrapper(numpy_array_api, 'wrapped_numpy.array_api')
    xp = _ArrayAPIWrapper(xp_)
    X = xp.asarray([[1, 2, 3], [3, 4, 5]], dtype=xp.float64)
    X_converted = xp.astype(X, xp.float32)
    assert X_converted.dtype == xp.float32
    X_converted = xp.asarray(X, dtype=xp.float32)
    assert X_converted.dtype == xp.float32
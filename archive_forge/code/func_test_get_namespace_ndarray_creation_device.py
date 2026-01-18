from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
def test_get_namespace_ndarray_creation_device():
    """Check expected behavior with device and creation functions."""
    X = numpy.asarray([1, 2, 3])
    xp_out, _ = get_namespace(X)
    full_array = xp_out.full(10, fill_value=2.0, device='cpu')
    assert_allclose(full_array, [2.0] * 10)
    with pytest.raises(ValueError, match='Unsupported device'):
        xp_out.zeros(10, device='cuda')
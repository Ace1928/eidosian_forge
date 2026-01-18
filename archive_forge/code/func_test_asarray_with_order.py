from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
@pytest.mark.parametrize('array_api', ['numpy', 'numpy.array_api'])
def test_asarray_with_order(array_api):
    """Test _asarray_with_order passes along order for NumPy arrays."""
    xp = pytest.importorskip(array_api)
    X = xp.asarray([1.2, 3.4, 5.1])
    X_new = _asarray_with_order(X, order='F', xp=xp)
    X_new_np = numpy.asarray(X_new)
    assert X_new_np.flags['F_CONTIGUOUS']
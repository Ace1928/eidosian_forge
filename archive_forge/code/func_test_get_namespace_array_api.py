from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
@skip_if_array_api_compat_not_configured
def test_get_namespace_array_api():
    """Test get_namespace for ArrayAPI arrays."""
    xp = pytest.importorskip('numpy.array_api')
    X_np = numpy.asarray([[1, 2, 3]])
    X_xp = xp.asarray(X_np)
    with config_context(array_api_dispatch=True):
        xp_out, is_array_api_compliant = get_namespace(X_xp)
        assert is_array_api_compliant
        assert isinstance(xp_out, _ArrayAPIWrapper)
        with pytest.raises(TypeError):
            xp_out, is_array_api_compliant = get_namespace(X_xp, X_np)
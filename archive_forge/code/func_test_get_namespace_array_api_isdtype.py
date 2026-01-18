from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
@pytest.mark.parametrize('wrapper', [_ArrayAPIWrapper, _NumPyAPIWrapper])
def test_get_namespace_array_api_isdtype(wrapper):
    """Test isdtype implementation from _ArrayAPIWrapper and _NumPyAPIWrapper."""
    if wrapper == _ArrayAPIWrapper:
        xp_ = pytest.importorskip('numpy.array_api')
        xp = _ArrayAPIWrapper(xp_)
    else:
        xp = _NumPyAPIWrapper()
    assert xp.isdtype(xp.float32, xp.float32)
    assert xp.isdtype(xp.float32, 'real floating')
    assert xp.isdtype(xp.float64, 'real floating')
    assert not xp.isdtype(xp.int32, 'real floating')
    for dtype in supported_float_dtypes(xp):
        assert xp.isdtype(dtype, 'real floating')
    assert xp.isdtype(xp.bool, 'bool')
    assert not xp.isdtype(xp.float32, 'bool')
    assert xp.isdtype(xp.int16, 'signed integer')
    assert not xp.isdtype(xp.uint32, 'signed integer')
    assert xp.isdtype(xp.uint16, 'unsigned integer')
    assert not xp.isdtype(xp.int64, 'unsigned integer')
    assert xp.isdtype(xp.int64, 'numeric')
    assert xp.isdtype(xp.float32, 'numeric')
    assert xp.isdtype(xp.uint32, 'numeric')
    assert not xp.isdtype(xp.float32, 'complex floating')
    if wrapper == _NumPyAPIWrapper:
        assert not xp.isdtype(xp.int8, 'complex floating')
        assert xp.isdtype(xp.complex64, 'complex floating')
        assert xp.isdtype(xp.complex128, 'complex floating')
    with pytest.raises(ValueError, match='Unrecognized data type'):
        assert xp.isdtype(xp.int16, 'unknown')
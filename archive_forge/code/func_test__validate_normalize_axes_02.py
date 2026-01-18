from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test__validate_normalize_axes_02():
    i, o = _validate_normalize_axes(None, 0, False, [('i',), ('i',)], ())
    assert i == [(0,), (0,)]
    assert o == [()]
    i, o = _validate_normalize_axes(None, 0, False, [('i',)], ('i',))
    assert i == [(0,)]
    assert o == [(0,)]
    i, o = _validate_normalize_axes(None, 0, True, [('i',), ('i',)], ())
    assert i == [(0,), (0,)]
    assert o == [(0,)]
    with pytest.raises(ValueError):
        _validate_normalize_axes(None, (0,), False, [('i',), ('i',)], ())
    with pytest.raises(ValueError):
        _validate_normalize_axes(None, 0, False, [('i',), ('j',)], ())
    with pytest.raises(ValueError):
        _validate_normalize_axes(None, 0, False, [('i',), ('j',)], ('j',))
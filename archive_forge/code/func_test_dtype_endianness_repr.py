import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
@pytest.mark.parametrize(['native'], [('bool',), ('uint8',), ('uint16',), ('uint32',), ('uint64',), ('int8',), ('int16',), ('int32',), ('int64',), ('float16',), ('float32',), ('float64',), ('U1',)])
def test_dtype_endianness_repr(self, native):
    """
        there was an issue where
        repr(array([0], dtype='<u2')) and repr(array([0], dtype='>u2'))
        both returned the same thing:
        array([0], dtype=uint16)
        even though their dtypes have different endianness.
        """
    native_dtype = np.dtype(native)
    non_native_dtype = native_dtype.newbyteorder()
    non_native_repr = repr(np.array([1], non_native_dtype))
    native_repr = repr(np.array([1], native_dtype))
    assert ('dtype' in native_repr) ^ (native_dtype in _typelessdata), "an array's repr should show dtype if and only if the type of the array is NOT one of the standard types (e.g., int32, bool, float64)."
    if non_native_dtype.itemsize > 1:
        assert non_native_repr != native_repr
        assert f"dtype='{non_native_dtype.byteorder}" in non_native_repr
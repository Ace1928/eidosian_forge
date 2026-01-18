import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
@pytest.mark.parametrize('dtype', np.typecodes['All'])
@pytest.mark.parametrize('loop_dtype', np.typecodes['All'])
@pytest.mark.filterwarnings('ignore::numpy.ComplexWarning')
def test_iter_copy_casts(dtype, loop_dtype):
    if loop_dtype.lower() == 'm':
        loop_dtype = loop_dtype + '[ms]'
    elif np.dtype(loop_dtype).itemsize == 0:
        loop_dtype = loop_dtype + '50'
    arr = np.ones(1000, dtype=np.dtype(dtype).newbyteorder())
    try:
        expected = arr.astype(loop_dtype)
    except Exception:
        return
    it = np.nditer((arr,), ['buffered', 'external_loop', 'refs_ok'], op_dtypes=[loop_dtype], casting='unsafe')
    if np.issubdtype(np.dtype(loop_dtype), np.number):
        assert_array_equal(expected, np.ones(1000, dtype=loop_dtype))
    it_copy = it.copy()
    res = next(it)
    del it
    res_copy = next(it_copy)
    del it_copy
    assert_array_equal(res, expected)
    assert_array_equal(res_copy, expected)
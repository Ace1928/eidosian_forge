import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_copy_casts_structured2():
    in_dtype = np.dtype([('a', np.dtype('O,O')), ('b', np.dtype('(5)O,(3)O,(1,)O,(1,)i,(1,)O'))])
    out_dtype = np.dtype([('a', np.dtype('O')), ('b', np.dtype('O,(3)i,(4)O,(4)O,(4)i'))])
    arr = np.ones(1, dtype=in_dtype)
    it = np.nditer((arr,), ['buffered', 'external_loop', 'refs_ok'], op_dtypes=[out_dtype], casting='unsafe')
    it_copy = it.copy()
    res1 = next(it)
    del it
    res2 = next(it_copy)
    del it_copy
    for res in (res1, res2):
        assert type(res['a'][0]) == tuple
        assert res['a'][0] == (1, 1)
    for res in (res1, res2):
        assert_array_equal(res['b']['f0'][0], np.ones(5, dtype=object))
        assert_array_equal(res['b']['f1'], np.ones((1, 3), dtype='i'))
        assert res['b']['f2'].shape == (1, 4)
        assert_array_equal(res['b']['f2'][0], np.ones(4, dtype=object))
        assert_array_equal(res['b']['f3'][0], np.ones(4, dtype=object))
        assert_array_equal(res['b']['f3'][0], np.ones(4, dtype='i'))
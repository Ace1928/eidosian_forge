import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def test_getitem_idx(self):
    nbarr = np.recarray(2, dtype=recordwitharray)
    nbarr[0] = np.array([(1, (2, 3))], dtype=recordwitharray)[0]
    for arg, retty in [(nbarr, recordwitharray), (nbarr[0], np.int32)]:
        pyfunc = recarray_getitem_return
        arr_expected = pyfunc(arg)
        cfunc = self.get_cfunc(pyfunc, retty)
        arr_res = cfunc(arg)
        np.testing.assert_equal(arr_res, arr_expected)
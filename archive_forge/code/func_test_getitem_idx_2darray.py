import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
@unittest.expectedFailure
def test_getitem_idx_2darray(self):
    nbarr = np.recarray(2, dtype=recordwith2darray)
    nbarr[0] = np.array([(1, ((1, 2), (4, 5), (2, 3)))], dtype=recordwith2darray)[0]
    for arg, retty in [(nbarr, recordwith2darray), (nbarr[0], (np.float32, (3, 2)))]:
        pyfunc = recarray_getitem_field_return2_2d
        arr_expected = pyfunc(arg)
        cfunc = self.get_cfunc(pyfunc, retty)
        arr_res = cfunc(arg)
        np.testing.assert_equal(arr_res, arr_expected)
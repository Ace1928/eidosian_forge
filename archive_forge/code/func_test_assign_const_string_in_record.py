import re
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from llvmlite import ir
def test_assign_const_string_in_record(self):

    @cuda.jit
    def f(a):
        a[0]['x'] = 1
        a[0]['y'] = 'ABC'
        a[1]['x'] = 2
        a[1]['y'] = 'XYZ'
    dt = np.dtype([('x', np.int32), ('y', np.dtype('<U12'))])
    a = np.zeros(2, dt)
    f[1, 1](a)
    reference = np.asarray([(1, 'ABC'), (2, 'XYZ')], dtype=dt)
    np.testing.assert_array_equal(reference, a)
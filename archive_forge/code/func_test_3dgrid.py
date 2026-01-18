import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_3dgrid(self):

    @cuda.jit
    def foo(out):
        x, y, z = cuda.grid(3)
        a, b, c = cuda.gridsize(3)
        out[x, y, z] = a * b * c
    arr = np.zeros(9 ** 3, dtype=np.int32).reshape(9, 9, 9)
    foo[(3, 3, 3), (3, 3, 3)](arr)
    np.testing.assert_equal(arr, 9 ** 3)
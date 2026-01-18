import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_3dgrid_2(self):

    @cuda.jit
    def foo(out):
        x, y, z = cuda.grid(3)
        a, b, c = cuda.gridsize(3)
        grid_is_right = x == cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x and y == cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y and (z == cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z)
        gridsize_is_right = a == cuda.blockDim.x * cuda.gridDim.x and b == cuda.blockDim.y * cuda.gridDim.y and (c == cuda.blockDim.z * cuda.gridDim.z)
        out[x, y, z] = grid_is_right and gridsize_is_right
    x, y, z = (4 * 3, 3 * 2, 2 * 4)
    arr = np.zeros(x * y * z, dtype=np.bool_).reshape(x, y, z)
    foo[(4, 3, 2), (3, 2, 4)](arr)
    self.assertTrue(np.all(arr))
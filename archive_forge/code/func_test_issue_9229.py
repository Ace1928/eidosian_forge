import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_on_cudasim('Requires too many threads')
def test_issue_9229(self):

    @cuda.jit
    def f(grid_error, gridsize_error):
        i1 = cuda.grid(1)
        i2 = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        gs1 = cuda.gridsize(1)
        gs2 = cuda.blockDim.x * cuda.gridDim.x
        if i1 != i2:
            grid_error[0] = 1
        if gs1 != gs2:
            gridsize_error[0] = 1
    grid_error = np.zeros(1, dtype=np.uint64)
    gridsize_error = np.zeros(1, dtype=np.uint64)
    f[22121216, 256](grid_error, gridsize_error)
    self.assertEqual(grid_error[0], 0)
    self.assertEqual(gridsize_error[0], 0)
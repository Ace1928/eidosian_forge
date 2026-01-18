import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_fill_threadidx(self):
    compiled = cuda.jit('void(int32[:])')(fill_threadidx)
    N = 10
    ary = np.ones(N, dtype=np.int32)
    exp = np.arange(N, dtype=np.int32)
    compiled[1, N](ary)
    self.assertTrue(np.all(ary == exp))
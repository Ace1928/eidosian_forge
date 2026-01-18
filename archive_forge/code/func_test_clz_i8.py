import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_on_cudasim('only get given a Python "int", assumes 32 bits')
def test_clz_i8(self):
    compiled = cuda.jit('void(int32[:], int64)')(simple_clz)
    ary = np.zeros(1, dtype=np.int32)
    compiled[1, 1](ary, 65536)
    self.assertEqual(ary[0], 47)
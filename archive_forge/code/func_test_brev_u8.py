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
def test_brev_u8(self):
    compiled = cuda.jit('void(uint64[:], uint64)')(simple_brev)
    ary = np.zeros(1, dtype=np.uint64)
    compiled[1, 1](ary, 53807350296816)
    self.assertEqual(ary[0], 1084241610541891584)
import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_brev_u4(self):
    compiled = cuda.jit('void(uint32[:], uint32)')(simple_brev)
    ary = np.zeros(1, dtype=np.uint32)
    compiled[1, 1](ary, 12528)
    self.assertEqual(ary[0], 252444672)
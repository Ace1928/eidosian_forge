import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_unless_cc_53
def test_habs(self):
    compiled = cuda.jit()(simple_habs)
    ary = np.zeros(1, dtype=np.float16)
    arg1 = np.array([-3.0], dtype=np.float16)
    compiled[1, 1](ary, arg1)
    np.testing.assert_allclose(ary[0], abs(arg1))
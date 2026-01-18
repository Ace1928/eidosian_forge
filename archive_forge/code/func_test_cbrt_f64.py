import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_cbrt_f64(self):
    compiled = cuda.jit('void(float64[:], float64)')(simple_cbrt)
    ary = np.zeros(1, dtype=np.float64)
    cbrt_arg = 6.0
    compiled[1, 1](ary, cbrt_arg)
    np.testing.assert_allclose(ary[0], cbrt_arg ** (1 / 3))
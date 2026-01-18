import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_multiple_hcmp_3(r, a, b, c):
    r[0] = hlt_func_1(a, b) and cuda.fp16.hge(c, b)
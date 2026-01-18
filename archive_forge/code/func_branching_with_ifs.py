import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def branching_with_ifs(a, b, c):
    i = cuda.grid(1)
    if a[i] > 4:
        if b % 2 == 0:
            a[i] = c[i]
        else:
            a[i] = 13
    else:
        a[i] = 3
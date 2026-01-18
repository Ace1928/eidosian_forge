import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def simple_hsqrt(r, x):
    i = cuda.grid(1)
    if i < len(r):
        r[i] = cuda.fp16.hsqrt(x[i])
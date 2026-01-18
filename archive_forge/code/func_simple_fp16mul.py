import numpy as np
from numba.cuda.testing import (unittest, CUDATestCase, skip_unless_cc_53,
from numba import cuda
from numba.core.types import f2, b1
from numba.cuda import compile_ptx
import operator
import itertools
from numba.np.numpy_support import from_dtype
def simple_fp16mul(ary, a, b):
    ary[0] = a * b
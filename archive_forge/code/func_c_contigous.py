import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def c_contigous():
    compiled = cuda.jit('void(int32[:,:,::1])')(fill3d_threadidx)
    ary = np.zeros((X, Y, Z), dtype=np.int32)
    compiled[1, (X, Y, Z)](ary)
    return ary
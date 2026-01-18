import re
import types
import numpy as np
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, float32, int32
from numba.core.errors import TypingError
@cuda.jit
def add_kernel(ary):
    i = cuda.grid(1)
    ary[i] = mymod.add(ary[i], 1)
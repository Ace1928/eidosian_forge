from __future__ import print_function
import numpy as np
from numba import config, cuda, int32
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@cuda.jit
def sync_group(A):
    g = cuda.cg.this_grid()
    g.sync()
    A[0] = 1.0
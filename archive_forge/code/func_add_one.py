import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
@cuda.jit
def add_one(arr):
    x = cuda.grid(1)
    N = arr.shape[0]
    if x < N:
        arr[x] += 1
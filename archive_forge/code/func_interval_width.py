from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import numpy as np
from numba import config, cuda, njit, types
@njit
def interval_width(interval):
    return interval.width
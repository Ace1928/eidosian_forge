import numpy as np
from numba.core import config
from numba.cuda.testing import CUDATestCase
from numba import cuda
def kernel_4elem(res):
    v = vobj(base_type(0), base_type(1), base_type(2), base_type(3))
    res[0] = v.x
    res[1] = v.y
    res[2] = v.z
    res[3] = v.w
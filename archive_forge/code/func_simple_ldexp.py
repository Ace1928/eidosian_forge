import numpy as np
import math
from numba import cuda
from numba.types import float32, float64, int32, void
from numba.cuda.testing import unittest, CUDATestCase
def simple_ldexp(aryx, arg, exp):
    aryx[0] = math.ldexp(arg, exp)
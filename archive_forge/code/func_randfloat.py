import numpy as np
import math
from numba import cuda, double, void
from numba.cuda.testing import unittest, CUDATestCase
def randfloat(rand_var, low, high):
    return (1.0 - rand_var) * low + rand_var * high
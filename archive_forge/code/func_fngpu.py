import numpy as np
from numba import vectorize, cuda
from numba.tests.npyufunc.test_vectorize_decor import BaseVectorizeDecor, \
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
@vectorize(['float64(float64,float64)'], target='cuda')
def fngpu(a, b):
    return a - b
import pickle
import numpy as np
from numba import cuda, vectorize
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
@vectorize(['intp(intp)', 'float64(float64)'], target='cuda')
def cuda_vect(x):
    return x * 2
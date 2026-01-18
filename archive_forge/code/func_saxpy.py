import numpy as np
from numba import guvectorize, cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
@guvectorize(['void(float32, float32[:], float32[:], float32[:])'], '(),(t),(t)->(t)', target='cuda')
def saxpy(a, x, y, out):
    for i in range(out.shape[0]):
        out[i] = a * x[i] + y[i]
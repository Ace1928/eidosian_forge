import numpy as np
from numba import guvectorize, cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
@guvectorize(['void(int32[:],int32[:],int32[:])'], '(n),()->(n)', target='cuda')
def gufunc(x, y, res):
    for i in range(x.shape[0]):
        res[i] = x[i] + y[0]
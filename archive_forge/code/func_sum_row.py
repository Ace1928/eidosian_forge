import numpy as np
from numba import guvectorize, cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
@guvectorize(['void(int32[:], int32[:])'], '(n)->()', target='cuda')
def sum_row(inp, out):
    tmp = 0.0
    for i in range(inp.shape[0]):
        tmp += inp[i]
    out[0] = tmp
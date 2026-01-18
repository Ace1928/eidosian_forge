import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase
import unittest
def reinterpret_array_type(byte_arr, start, stop, output):
    val = byte_arr[start:stop].view(np.int32)[0]
    output[0] = val
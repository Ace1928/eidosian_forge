import re
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from llvmlite import ir
@cuda.jit
def bytes_assign(arr):
    i = cuda.grid(1)
    if i < len(arr):
        arr[i] = b'XYZ'
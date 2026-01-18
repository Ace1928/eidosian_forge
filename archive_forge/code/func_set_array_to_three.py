import numpy as np
from numba import cuda
from numba.cuda.args import wrap_arg
from numba.cuda.testing import CUDATestCase
import unittest
def set_array_to_three(arr):
    arr[0] = 3
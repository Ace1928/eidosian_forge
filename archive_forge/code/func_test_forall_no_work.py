import numpy as np
from numba import cuda
import unittest
from numba.cuda.testing import CUDATestCase
def test_forall_no_work(self):
    arr = np.arange(11)
    foo.forall(0)(arr)
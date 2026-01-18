import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_1d_view(self):
    shape = 10
    view = np.zeros(shape)[::2]
    self._test_against_array_core(view)
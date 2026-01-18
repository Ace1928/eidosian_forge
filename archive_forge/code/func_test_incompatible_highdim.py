from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_incompatible_highdim(self):
    darr = cuda.to_device(np.arange(5 * 7))
    with self.assertRaises(ValueError) as e:
        darr[:] = np.ones(shape=(1, 2, 3))
    self.assertIn(member=str(e.exception), container=["Can't assign 3-D array to 1-D self", 'could not broadcast input array from shape (2,3) into shape (35,)'])
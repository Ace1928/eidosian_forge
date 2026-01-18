from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_incompatible_shape(self):
    darr = cuda.to_device(np.arange(5))
    with self.assertRaises(ValueError) as e:
        darr[:] = [1, 3]
    self.assertIn(member=str(e.exception), container=["Can't copy sequence with size 2 to array axis 0 with dimension 5", 'could not broadcast input array from shape (2,) into shape (5,)'])
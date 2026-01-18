from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_index_3d(self):
    arr = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    darr = cuda.to_device(arr)
    x, y, z = arr.shape
    for i in range(-x, x):
        for j in range(-y, y):
            for k in range(-z, z):
                self.assertEqual(arr[i, j, k], darr[i, j, k])
    with self.assertRaises(IndexError):
        darr[-x - 1, 0, 0]
    with self.assertRaises(IndexError):
        darr[x, 0, 0]
    with self.assertRaises(IndexError):
        darr[0, -y - 1, 0]
    with self.assertRaises(IndexError):
        darr[0, y, 0]
    with self.assertRaises(IndexError):
        darr[0, 0, -z - 1]
    with self.assertRaises(IndexError):
        darr[0, 0, z]
import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
def test_extent_iter_2d(self):
    nparr = np.empty((4, 5))
    arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
    [ext] = list(arr.iter_contiguous_extent())
    self.assertEqual(ext, arr.extent)
    self.assertEqual(len(list(arr[::2].iter_contiguous_extent())), 2)
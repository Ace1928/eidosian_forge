import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
def test_extent_iter_1d(self):
    nparr = np.empty(4)
    arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
    [ext] = list(arr.iter_contiguous_extent())
    self.assertEqual(ext, arr.extent)
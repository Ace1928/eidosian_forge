import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_fully_described(self):
    mbslice = MultiBlockSlice(start=1, count=2, stride=5, block=4)
    self.assertEqual(mbslice.indices(10), (1, 5, 2, 4))
    np.testing.assert_array_equal(self.dset[mbslice], np.array([1, 2, 3, 4, 6, 7, 8, 9]))
from ..common import ut
import h5py as h5
import numpy as np
def test_double_strided_range(self):
    dataset = h5.VirtualSource('test', 'test', (20, 30, 30))
    sliced = dataset[6:12:2, :, 20:26:3]
    self.assertEqual((3, 30, 2), sliced.shape)
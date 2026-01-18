from ..common import ut
import h5py as h5
import numpy as np
def test_negative_start_index(self):
    dataset = h5.VirtualSource('test', 'test', (20, 30, 30))
    sliced = dataset[-10:16]
    self.assertEqual((6, 30, 30), sliced.shape)
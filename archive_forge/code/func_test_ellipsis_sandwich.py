from ..common import ut
import h5py as h5
import numpy as np
def test_ellipsis_sandwich(self):
    dataset = h5.VirtualSource('test', 'test', (20, 30, 30, 40))
    sliced = dataset[0:1, ..., 5:6]
    self.assertEqual((1,) + dataset.shape[1:-1] + (1,), sliced.shape)
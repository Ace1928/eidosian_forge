from ..common import ut
import h5py as h5
import numpy as np
def test_ellipsis_start(self):
    dataset = h5.VirtualSource('test', 'test', (20, 30, 30))
    sliced = dataset[..., 0:1]
    self.assertEqual(dataset.shape[:-1] + (1,), sliced.shape)
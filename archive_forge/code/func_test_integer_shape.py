from ..common import ut
import h5py as h5
import numpy as np
def test_integer_shape(self):
    dataset = h5.VirtualSource('test', 'test', 20)
    self.assertEqual(dataset.shape, (20,))
from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_b16_uint16(self):
    arr1 = np.arange(10, dtype=np.uint16)
    path = self.mktemp()
    with h5py.File(path, 'w') as f:
        space = h5py.h5s.create_simple(arr1.shape)
        dset_id = h5py.h5d.create(f.id, b'test', h5py.h5t.STD_B16LE, space)
        dset = h5py.Dataset(dset_id)
        dset[:] = arr1
    with h5py.File(path, 'r') as f:
        dset = f['test']
        self.assertArrayEqual(dset[:], arr1)
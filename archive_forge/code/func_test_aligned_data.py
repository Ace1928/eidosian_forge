from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_aligned_data(self):
    dt = np.dtype('i4,f8,i2', align=True)
    data = np.zeros(10, dtype=dt)
    data['f0'] = np.array(np.random.randint(-100, 100, size=data.size), dtype='i4')
    data['f1'] = np.random.rand(data.size)
    data['f2'] = np.array(np.random.randint(-100, 100, size=data.size), dtype='i2')
    fname = self.mktemp()
    with h5py.File(fname, 'w') as f:
        f['data'] = data
    with h5py.File(fname, 'r') as f:
        self.assertArrayEqual(f['data'], data)
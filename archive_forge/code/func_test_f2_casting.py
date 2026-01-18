from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_f2_casting(self):
    fname = self.mktemp()
    np.random.seed(1)
    A = np.random.rand(1500, 20)
    with h5py.File(fname, 'w') as Fid:
        Fid.create_dataset('Data', data=A, dtype='f2')
    with h5py.File(fname, 'r') as Fid:
        B = Fid['Data'][:]
    self.assertTrue(np.all(A.astype('f2') == B))
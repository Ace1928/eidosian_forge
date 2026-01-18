import numpy as np
from .common import TestCase, ut
import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace
def test_nesting_compound_with_vlen_fields(self):
    """ Compound scalars with nested compound vlen fields can be written and read """
    dt_inner = np.dtype([('a', h5py.vlen_dtype(np.int32)), ('b', h5py.vlen_dtype(np.int32))])
    dt = np.dtype([('f1', h5py.vlen_dtype(dt_inner)), ('f2', np.int64)])
    inner1 = (np.array(range(1, 3), dtype=np.int32), np.array(range(6, 9), dtype=np.int32))
    inner2 = (np.array(range(10, 14), dtype=np.int32), np.array(range(16, 20), dtype=np.int32))
    data = np.array((np.array([inner1, inner2], dtype=dt_inner), 2), dtype=dt)[()]
    self.f.attrs['x'] = data
    out = self.f.attrs['x']
    self.assertArrayEqual(out, data, check_alignment=False)
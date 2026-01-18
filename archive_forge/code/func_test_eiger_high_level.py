import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
def test_eiger_high_level(self):
    outfile = osp.join(self.working_dir, 'eiger.h5')
    layout = h5.VirtualLayout(shape=(78, 200, 200), dtype=float)
    M_minus_1 = 0
    with h5.File(outfile, 'w', libver='latest') as f:
        for foo in self.fname:
            in_data = h5.File(foo, 'r')['data']
            src_shape = in_data.shape
            in_data.file.close()
            M = M_minus_1 + src_shape[0]
            vsource = h5.VirtualSource(foo, 'data', shape=src_shape)
            layout[M_minus_1:M, :, :] = vsource
            M_minus_1 = M
        f.create_virtual_dataset('data', layout, fillvalue=45)
    f = h5.File(outfile, 'r')['data']
    self.assertEqual(f[10, 100, 10], 0.0)
    self.assertEqual(f[30, 100, 100], 1.0)
    self.assertEqual(f[50, 100, 100], 2.0)
    self.assertEqual(f[70, 100, 100], 3.0)
    f.file.close()
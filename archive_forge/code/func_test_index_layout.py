import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
def test_index_layout(self):
    layout = h5.VirtualLayout((100,), 'i4')
    inds = [3, 6, 20, 25, 33, 47, 70, 75, 96, 98]
    filename = osp.join(self.tmpdir, '1.h5')
    vsource = h5.VirtualSource(filename, 'data', shape=(10,))
    layout[inds] = vsource
    outfile = osp.join(self.tmpdir, 'VDS.h5')
    layout2 = h5.VirtualLayout((6,), 'i4')
    inds2 = [0, 1, 4, 5, 8]
    layout2[1:] = vsource[inds2]
    with h5.File(outfile, 'w', libver='latest') as f:
        f.create_virtual_dataset('/data', layout, fillvalue=-5)
        f.create_virtual_dataset(b'/data2', layout2, fillvalue=-3)
    with h5.File(outfile, 'r') as f:
        data = f['/data'][()]
        data2 = f['/data2'][()]
    assert_array_equal(data[inds], np.arange(10) * 10)
    assert_array_equal(data2[1:], [0, 10, 40, 50, 80])
    mask = np.zeros(100)
    mask[inds] = 1
    self.assertEqual(data[mask == 0].min(), -5)
    self.assertEqual(data[mask == 0].max(), -5)
    self.assertEqual(data2[0], -3)
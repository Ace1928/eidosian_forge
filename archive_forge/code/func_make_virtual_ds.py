import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
def make_virtual_ds(self):
    layout = h5.VirtualLayout((4, 100), 'i4', maxshape=(4, None))
    for n in range(1, 5):
        filename = osp.join(self.tmpdir, '{}.h5'.format(n))
        vsource = h5.VirtualSource(filename, 'data', shape=(100,))
        layout[n - 1, :50] = vsource[0:100:2]
        layout[n - 1, 50:] = vsource[1:100:2]
    outfile = osp.join(self.tmpdir, 'VDS.h5')
    with h5.File(outfile, 'w', libver='latest') as f:
        f.create_virtual_dataset('/group/data', layout, fillvalue=-5)
    return outfile
import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
def test_mismatched_selections(self):
    layout = h5.VirtualLayout((4, 100), 'i4', maxshape=(4, None))
    filename = osp.join(self.tmpdir, '1.h5')
    vsource = h5.VirtualSource(filename, 'data', shape=(100,))
    with self.assertRaisesRegex(ValueError, 'different number'):
        layout[0, :49] = vsource[0:100:2]
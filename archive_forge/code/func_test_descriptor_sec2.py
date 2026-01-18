import tempfile
import shutil
import os
import numpy as np
from h5py import File, special_dtype
from h5py._hl.files import direct_vfd
from .common import ut, TestCase
from .common import ut, TestCase, UNICODE_FILENAMES, closed_tempfile
def test_descriptor_sec2(self):
    dn_tmp = tempfile.mkdtemp('h5py.lowtest.test_h5f.TestFileID.test_descriptor_sec2')
    fn_h5 = os.path.join(dn_tmp, 'test.h5')
    try:
        with File(fn_h5, driver='sec2', mode='x') as f:
            descriptor = f.id.get_vfd_handle()
            self.assertNotEqual(descriptor, 0)
            os.fsync(descriptor)
    finally:
        shutil.rmtree(dn_tmp)
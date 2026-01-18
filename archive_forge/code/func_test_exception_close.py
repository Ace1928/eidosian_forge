import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
@ut.skip('Incompletely closed files can cause segfaults')
def test_exception_close(self):
    fileobj = io.BytesIO()
    f = h5py.File(fileobj, 'w')
    fileobj.close()
    self.assertRaises(Exception, f.close)
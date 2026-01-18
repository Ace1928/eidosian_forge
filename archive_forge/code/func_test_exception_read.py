import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
def test_exception_read(self):

    class BrokenBytesIO(io.BytesIO):

        def readinto(self, b):
            raise Exception('I am broken')
    f = h5py.File(BrokenBytesIO(), 'w')
    f.create_dataset('test', data=list(range(12)))
    self.assertRaises(Exception, list, f['test'])
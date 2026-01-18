import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
def test_exception_write(self):

    class BrokenBytesIO(io.BytesIO):
        allow_write = False

        def write(self, b):
            if self.allow_write:
                return super().write(b)
            else:
                raise Exception('I am broken')
    bio = BrokenBytesIO()
    f = h5py.File(bio, 'w')
    try:
        self.assertRaises(Exception, f.create_dataset, 'test', data=list(range(12)))
    finally:
        bio.allow_write = True
        f.close()
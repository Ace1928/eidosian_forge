import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
def test_exception_open(self):
    self.assertRaises(Exception, h5py.File, None, driver='fileobj', mode='x')
    self.assertRaises(Exception, h5py.File, 'rogue', driver='fileobj', mode='x')
    self.assertRaises(Exception, h5py.File, self, driver='fileobj', mode='x')
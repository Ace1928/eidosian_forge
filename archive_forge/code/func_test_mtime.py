import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_mtime(self):
    mtime = 123456789
    with gzip.GzipFile(self.filename, 'w', mtime=mtime) as fWrite:
        fWrite.write(data1)
    with gzip.GzipFile(self.filename) as fRead:
        dataRead = fRead.read()
        self.assertEqual(dataRead, data1)
        self.assertTrue(hasattr(fRead, 'mtime'))
        self.assertEqual(fRead.mtime, mtime)
import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_binary_modes(self):
    uncompressed = data1 * 50
    with gzip.open(self.filename, 'wb') as f:
        f.write(uncompressed)
    with open(self.filename, 'rb') as f:
        file_data = gzip.decompress(f.read())
        self.assertEqual(file_data, uncompressed)
    with gzip.open(self.filename, 'rb') as f:
        self.assertEqual(f.read(), uncompressed)
    with gzip.open(self.filename, 'ab') as f:
        f.write(uncompressed)
    with open(self.filename, 'rb') as f:
        file_data = gzip.decompress(f.read())
        self.assertEqual(file_data, uncompressed * 2)
import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_many_append(self):
    with gzip.GzipFile(self.filename, 'wb', 9) as f:
        f.write(b'a')
    for i in range(0, 200):
        with gzip.GzipFile(self.filename, 'ab', 9) as f:
            f.write(b'a')
    with gzip.GzipFile(self.filename, 'rb') as zgfile:
        contents = b''
        while 1:
            ztxt = zgfile.read(8192)
            contents += ztxt
            if not ztxt:
                break
    self.assertEqual(contents, b'a' * 201)
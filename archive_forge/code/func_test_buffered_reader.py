import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_buffered_reader(self):
    self.test_write()
    with gzip.GzipFile(self.filename, 'rb') as f:
        with io.BufferedReader(f) as r:
            lines = [line for line in r]
    self.assertEqual(lines, 50 * data1.splitlines(True))
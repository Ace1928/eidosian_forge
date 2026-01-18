import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_decompress(self):
    for data in (data1, data2):
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb') as f:
            f.write(data)
        self.assertEqual(gzip.decompress(buf.getvalue()), data)
        datac = gzip.compress(data)
        self.assertEqual(gzip.decompress(datac), data)
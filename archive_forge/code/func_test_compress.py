import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_compress(self):
    for data in [data1, data2]:
        for args in [(), (1,), (6,), (9,)]:
            datac = gzip.compress(data, *args)
            self.assertEqual(type(datac), bytes)
            with gzip.GzipFile(fileobj=io.BytesIO(datac), mode='rb') as f:
                self.assertEqual(f.read(), data)
import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_non_seekable_file(self):
    uncompressed = data1 * 50
    buf = UnseekableIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as f:
        f.write(uncompressed)
    compressed = buf.getvalue()
    buf = UnseekableIO(compressed)
    with gzip.GzipFile(fileobj=buf, mode='rb') as f:
        self.assertEqual(f.read(), uncompressed)
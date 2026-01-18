import unittest
from apitools.base.py import compression
from apitools.base.py import gzip
import six
def testCompressionIntegrity(self):
    """Test that compressed data can be decompressed."""
    output, read, exhausted = compression.CompressStream(self.stream, self.length, 9)
    with gzip.GzipFile(fileobj=output) as f:
        original = f.read()
        self.assertEqual(original, self.sample_data)
    self.assertEqual(read, self.length)
    self.assertTrue(exhausted)
import unittest
from apitools.base.py import compression
from apitools.base.py import gzip
import six
def testTooShort(self):
    """Test excessive stream reads.

        Test that more data can be requested from the stream than available
        without raising an exception.
        """
    self.stream.write(b'Sample')
    data = self.stream.read(100)
    self.assertEqual(data, b'Sample')
    self.assertEqual(self.stream.length, 0)
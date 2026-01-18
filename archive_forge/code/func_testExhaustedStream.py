import string
import unittest
import six
from apitools.base.py import buffered_stream
from apitools.base.py import exceptions
def testExhaustedStream(self):
    bs = buffered_stream.BufferedStream(self.stream, 0, 100)
    self.assertEqual(len(self.value), bs.stream_end_position)
    self.assertEqual(True, bs.stream_exhausted)
    self.assertEqual(self.value, bs.read(100))
    self.assertEqual('', bs.read(0))
    self.assertEqual('', bs.read(100))
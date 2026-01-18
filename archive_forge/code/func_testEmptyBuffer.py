import string
import unittest
import six
from apitools.base.py import buffered_stream
from apitools.base.py import exceptions
def testEmptyBuffer(self):
    bs = buffered_stream.BufferedStream(self.stream, 0, 0)
    self.assertEqual('', bs.read(0))
    self.assertEqual(0, bs.stream_end_position)
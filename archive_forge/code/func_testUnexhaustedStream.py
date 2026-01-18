import string
import unittest
import six
from apitools.base.py import buffered_stream
from apitools.base.py import exceptions
def testUnexhaustedStream(self):
    bs = buffered_stream.BufferedStream(self.stream, 0, 50)
    self.assertEqual(50, bs.stream_end_position)
    self.assertEqual(False, bs.stream_exhausted)
    self.assertEqual(self.value[0:50], bs.read(50))
    self.assertEqual(False, bs.stream_exhausted)
    self.assertEqual('', bs.read(0))
    self.assertEqual('', bs.read(100))
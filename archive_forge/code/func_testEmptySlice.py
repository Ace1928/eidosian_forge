import string
import unittest
import six
from apitools.base.py import exceptions
from apitools.base.py import stream_slice
def testEmptySlice(self):
    ss = stream_slice.StreamSlice(self.stream, 0)
    self.assertEqual('', ss.read(5))
    self.assertEqual('', ss.read())
    self.assertEqual(0, self.stream.tell())
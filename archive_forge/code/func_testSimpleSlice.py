import string
import unittest
import six
from apitools.base.py import exceptions
from apitools.base.py import stream_slice
def testSimpleSlice(self):
    ss = stream_slice.StreamSlice(self.stream, 10)
    self.assertEqual('', ss.read(0))
    self.assertEqual(self.value[0:3], ss.read(3))
    self.assertIn('7/10', str(ss))
    self.assertEqual(self.value[3:10], ss.read())
    self.assertEqual('', ss.read())
    self.assertEqual('', ss.read(10))
    self.assertEqual(10, self.stream.tell())
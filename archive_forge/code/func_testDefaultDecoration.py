import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testDefaultDecoration(self):

    @util.positional
    def fn(a, b, c=None):
        return (a, b, c)
    self.assertEquals((1, 2, 3), fn(1, 2, c=3))
    self.assertEquals((3, 4, None), fn(3, b=4))
    self.assertRaisesWithRegexpMatch(TypeError, 'fn\\(\\) takes at most 2 positional arguments \\(3 given\\)', fn, 2, 3, 4)
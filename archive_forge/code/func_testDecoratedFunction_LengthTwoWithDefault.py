import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testDecoratedFunction_LengthTwoWithDefault(self):

    @util.positional(2)
    def fn(pos1, pos2=1, kwonly=1):
        return [pos1, pos2, kwonly]
    self.assertEquals([1, 1, 1], fn(1))
    self.assertEquals([2, 2, 1], fn(2, 2))
    self.assertEquals([2, 3, 4], fn(2, 3, kwonly=4))
    self.assertRaisesWithRegexpMatch(TypeError, 'fn\\(\\) takes at most 2 positional arguments \\(3 given\\)', fn, 2, 3, 4)
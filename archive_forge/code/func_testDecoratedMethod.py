import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testDecoratedMethod(self):

    class MyClass(object):

        @util.positional(2)
        def meth(self, pos1, kwonly=1):
            return [pos1, kwonly]
    self.assertEquals([1, 1], MyClass().meth(1))
    self.assertEquals([2, 2], MyClass().meth(2, kwonly=2))
    self.assertRaisesWithRegexpMatch(TypeError, 'meth\\(\\) takes at most 2 positional arguments \\(3 given\\)', MyClass().meth, 2, 3)
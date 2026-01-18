import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testDecoratedFunctionDocstring(self):

    @util.positional(0)
    def fn(kwonly=1):
        """fn docstring."""
        return [kwonly]
    self.assertEquals('fn docstring.', fn.__doc__)
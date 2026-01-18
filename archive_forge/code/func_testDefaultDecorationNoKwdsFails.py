import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testDefaultDecorationNoKwdsFails(self):

    def fn(a):
        return a
    self.assertRaisesRegexp(ValueError, 'Functions with no keyword arguments must specify max_positional_args', util.positional, fn)
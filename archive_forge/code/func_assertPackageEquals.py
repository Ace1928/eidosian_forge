import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def assertPackageEquals(self, expected, actual):
    self.assertEquals(expected, actual)
    if actual is not None:
        self.assertTrue(isinstance(actual, six.text_type))
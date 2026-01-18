import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testClassNotMutable(self):
    """Test that enum classes themselves are not mutable."""
    self.assertRaises(AttributeError, setattr, Color, 'something_new', 10)
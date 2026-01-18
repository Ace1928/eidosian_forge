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
def testInstancesMutable(self):
    """Test that enum instances are not mutable."""
    self.assertRaises(TypeError, setattr, Color.RED, 'something_new', 10)
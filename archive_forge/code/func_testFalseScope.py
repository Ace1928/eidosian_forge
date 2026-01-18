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
def testFalseScope(self):
    """Test Message definitions nested in strange objects are hidden."""
    global X

    class X(object):

        class A(messages.Message):
            pass
    self.assertRaises(TypeError, messages.find_definition, 'A', X)
    self.assertRaises(messages.DefinitionNotFoundError, messages.find_definition, 'X.A', sys.modules[__name__])
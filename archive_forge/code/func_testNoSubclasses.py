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
def testNoSubclasses(self):
    """Test that it is not possible to sub-class enum classes."""

    def declare_subclass():

        class MoreColor(Color):
            pass
    self.assertRaises(messages.EnumDefinitionError, declare_subclass)
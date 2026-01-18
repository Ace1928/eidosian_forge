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
def testSubclassingMessageDisallowed(self):
    """Not permitted to create sub-classes of message classes."""

    class SuperClass(messages.Message):
        pass

    def action():

        class SubClass(SuperClass):
            pass
    self.assertRaises(messages.MessageDefinitionError, action)
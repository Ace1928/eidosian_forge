import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testField(self):
    self.assertEquals(descriptor.describe_field(test_util.NestedMessage.a_value), descriptor.describe(test_util.NestedMessage.a_value))
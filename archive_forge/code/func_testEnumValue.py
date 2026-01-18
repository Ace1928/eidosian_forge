import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testEnumValue(self):
    self.assertEquals(descriptor.describe_enum_value(test_util.OptionalMessage.SimpleEnum.VAL1), descriptor.describe(test_util.OptionalMessage.SimpleEnum.VAL1))
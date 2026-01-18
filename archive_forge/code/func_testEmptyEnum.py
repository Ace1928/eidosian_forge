import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testEmptyEnum(self):

    class EmptyEnum(messages.Enum):
        pass
    expected = descriptor.EnumDescriptor()
    expected.name = 'EmptyEnum'
    described = descriptor.describe_enum(EmptyEnum)
    described.check_initialized()
    self.assertEquals(expected, described)
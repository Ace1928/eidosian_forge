import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testNoPackageName(self):
    """Test describing a module with no module name."""
    module = types.ModuleType('')
    expected = descriptor.FileDescriptor()
    described = descriptor.describe_file(module)
    described.check_initialized()
    self.assertEquals(expected, described)
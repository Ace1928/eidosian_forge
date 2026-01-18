import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testPackageName(self):
    """Test using the 'package' module attribute."""
    module = types.ModuleType('my.module.name')
    module.package = 'my.package.name'
    expected = descriptor.FileDescriptor()
    expected.package = 'my.package.name'
    described = descriptor.describe_file(module)
    described.check_initialized()
    self.assertEquals(expected, described)
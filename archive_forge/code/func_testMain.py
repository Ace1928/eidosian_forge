import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testMain(self):
    """Test using the 'package' module attribute."""
    module = types.ModuleType('__main__')
    module.__file__ = '/blim/blam/bloom/my_package.py'
    expected = descriptor.FileDescriptor()
    expected.package = 'my_package'
    described = descriptor.describe_file(module)
    described.check_initialized()
    self.assertEquals(expected, described)
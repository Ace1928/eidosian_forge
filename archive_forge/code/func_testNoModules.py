import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testNoModules(self):
    """Test what happens when no modules provided."""
    described = descriptor.describe_file_set([])
    described.check_initialized()
    self.assertEquals(descriptor.FileSet(), described)
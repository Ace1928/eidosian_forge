import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testFindField(self):
    self.assertEquals(descriptor.describe_field(descriptor.FileSet.files), descriptor.import_descriptor_loader('apitools.base.protorpclite.descriptor.FileSet.files'))
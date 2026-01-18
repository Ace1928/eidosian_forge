import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testLookupPackage(self):
    self.assertEquals('csv', self.library.lookup_package('csv'))
    self.assertEquals('apitools.base.protorpclite', self.library.lookup_package('apitools.base.protorpclite'))
import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testNoPackage(self):
    self.assertRaisesWithRegexpMatch(messages.DefinitionNotFoundError, 'Could not find definition for not.real', self.library.lookup_package, 'not.real.Packageless')
    self.assertEquals(None, self.library.lookup_package('Packageless'))
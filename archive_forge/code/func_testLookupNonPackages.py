import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testLookupNonPackages(self):
    lib = 'apitools.base.protorpclite.descriptor.DescriptorLibrary'
    for name in ('', 'a', lib):
        self.assertRaisesWithRegexpMatch(messages.DefinitionNotFoundError, 'Could not find definition for %s' % name, self.library.lookup_package, name)
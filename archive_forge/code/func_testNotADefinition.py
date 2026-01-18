import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testNotADefinition(self):
    """Test trying to fetch something that is not a definition."""
    module = self.DefineModule('i.am.a.module')
    setattr(module, 'A', 'a string')
    self.assertRaises(messages.DefinitionNotFoundError, messages.find_definition, 'i.am.a.module.A', importer=self.Importer)
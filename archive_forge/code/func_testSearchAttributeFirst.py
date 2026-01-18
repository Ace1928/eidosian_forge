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
def testSearchAttributeFirst(self):
    """Make sure not faked out by module, but continues searching."""
    A = self.DefineMessage('a', 'A')
    module_A = self.DefineModule('a.A')
    self.assertEquals(A, messages.find_definition('a.A', None, importer=self.Importer))
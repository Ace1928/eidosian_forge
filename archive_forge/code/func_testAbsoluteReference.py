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
def testAbsoluteReference(self):
    """Test finding absolute definition names."""
    a = self.DefineModule('a')
    b = self.DefineModule('a.a')
    aA = self.DefineMessage('a', 'A')
    aaA = self.DefineMessage('a.a', 'A')
    self.assertEquals(aA, messages.find_definition('.a.A', None, importer=self.Importer))
    self.assertEquals(aA, messages.find_definition('.a.A', a, importer=self.Importer))
    self.assertEquals(aA, messages.find_definition('.a.A', aA, importer=self.Importer))
    self.assertEquals(aA, messages.find_definition('.a.A', aaA, importer=self.Importer))
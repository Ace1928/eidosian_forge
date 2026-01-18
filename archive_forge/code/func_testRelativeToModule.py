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
def testRelativeToModule(self):
    """Test finding definitions relative to modules."""
    a = self.DefineModule('a')
    b = self.DefineModule('a.b')
    c = self.DefineModule('a.b.c')
    A = self.DefineMessage('a', 'A')
    B = self.DefineMessage('a.b', 'B')
    C = self.DefineMessage('a.b.c', 'C')
    D = self.DefineMessage('a.b.d', 'D')
    self.assertEquals(A, messages.find_definition('A', a, importer=self.Importer))
    self.assertEquals(B, messages.find_definition('b.B', a, importer=self.Importer))
    self.assertEquals(C, messages.find_definition('b.c.C', a, importer=self.Importer))
    self.assertEquals(D, messages.find_definition('b.d.D', a, importer=self.Importer))
    self.assertEquals(A, messages.find_definition('A', b, importer=self.Importer))
    self.assertEquals(B, messages.find_definition('B', b, importer=self.Importer))
    self.assertEquals(C, messages.find_definition('c.C', b, importer=self.Importer))
    self.assertEquals(D, messages.find_definition('d.D', b, importer=self.Importer))
    self.assertEquals(A, messages.find_definition('A', c, importer=self.Importer))
    self.assertEquals(B, messages.find_definition('B', c, importer=self.Importer))
    self.assertEquals(C, messages.find_definition('C', c, importer=self.Importer))
    self.assertEquals(D, messages.find_definition('d.D', c, importer=self.Importer))
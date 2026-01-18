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
def testAllFields(self):
    """Test all_fields method."""
    ComplexMessage = self.CreateMessageClass()
    fields = list(ComplexMessage.all_fields())
    fields = sorted(fields, key=lambda f: f.name)
    self.assertEquals(3, len(fields))
    self.assertEquals('a3', fields[0].name)
    self.assertEquals('b1', fields[1].name)
    self.assertEquals('c2', fields[2].name)
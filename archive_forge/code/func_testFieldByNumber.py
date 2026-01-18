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
def testFieldByNumber(self):
    """Test getting field by number."""
    ComplexMessage = self.CreateMessageClass()
    self.assertEquals('a3', ComplexMessage.field_by_number(3).name)
    self.assertEquals('b1', ComplexMessage.field_by_number(1).name)
    self.assertEquals('c2', ComplexMessage.field_by_number(2).name)
    self.assertRaises(KeyError, ComplexMessage.field_by_number, 4)
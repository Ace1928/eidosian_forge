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
def testConstructor(self):
    self.assertEquals([1, 2, 3], messages.FieldList(self.integer_field, [1, 2, 3]))
    self.assertEquals([1, 2, 3], messages.FieldList(self.integer_field, (1, 2, 3)))
    self.assertEquals([], messages.FieldList(self.integer_field, []))
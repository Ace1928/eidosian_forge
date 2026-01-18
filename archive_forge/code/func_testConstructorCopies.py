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
def testConstructorCopies(self):
    a_list = [1, 3, 6]
    field_list = messages.FieldList(self.integer_field, a_list)
    self.assertFalse(a_list is field_list)
    self.assertFalse(field_list is messages.FieldList(self.integer_field, field_list))
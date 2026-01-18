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
def testConstructor_Scalars(self):
    self.assertRaisesWithRegexpMatch(messages.ValidationError, 'IntegerField is repeated. Found: 3', messages.FieldList, self.integer_field, 3)
    self.assertRaisesWithRegexpMatch(messages.ValidationError, 'IntegerField is repeated. Found: <(list[_]?|sequence)iterator object', messages.FieldList, self.integer_field, iter([1, 2, 3]))
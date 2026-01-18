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
def testNonRepeatedField(self):
    self.assertRaisesWithRegexpMatch(messages.FieldDefinitionError, 'FieldList may only accept repeated fields', messages.FieldList, messages.IntegerField(1), [])
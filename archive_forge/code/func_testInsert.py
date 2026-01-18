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
def testInsert(self):
    field_list = messages.FieldList(self.integer_field, [2, 3])
    field_list.insert(1, 10)
    self.assertEquals([2, 10, 3], field_list)
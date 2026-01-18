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
def testAppend_InvalidValues(self):
    field_list = messages.FieldList(self.integer_field, [2])
    field_list.name = 'a_field'

    def append():
        field_list.append('10')
    self.assertRaisesWithRegexpMatch(messages.ValidationError, re.escape('Expected type %r for IntegerField, found 10 (type %r)' % (six.integer_types, str)), append)
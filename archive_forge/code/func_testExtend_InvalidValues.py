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
def testExtend_InvalidValues(self):
    field_list = messages.FieldList(self.integer_field, [2])

    def extend():
        field_list.extend(['10'])
    self.assertRaisesWithRegexpMatch(messages.ValidationError, re.escape('Expected type %r for IntegerField, found 10 (type %r)' % (six.integer_types, str)), extend)
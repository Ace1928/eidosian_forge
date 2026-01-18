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
def testSetSlice_InvalidValues(self):
    field_list = messages.FieldList(self.integer_field, [1, 2, 3, 4, 5])

    def setslice():
        field_list[1:3] = ['10', '20']
    msg_re = re.escape('Expected type %r for IntegerField, found 10 (type %r)' % (six.integer_types, str))
    self.assertRaisesWithRegexpMatch(messages.ValidationError, msg_re, setslice)
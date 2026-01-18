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
def testGetAssignedValue(self):
    """Test getting the assigned value of a field."""

    class SomeMessage(messages.Message):
        a_value = messages.StringField(1, default=u'a default')
    message = SomeMessage()
    self.assertEquals(None, message.get_assigned_value('a_value'))
    message.a_value = u'a string'
    self.assertEquals(u'a string', message.get_assigned_value('a_value'))
    message.a_value = u'a default'
    self.assertEquals(u'a default', message.get_assigned_value('a_value'))
    self.assertRaisesWithRegexpMatch(AttributeError, 'Message SomeMessage has no field no_such_field', message.get_assigned_value, 'no_such_field')
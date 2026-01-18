import datetime
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testValueFromMessageBadValue(self):
    field = message_types.DateTimeField(1)
    self.assertRaisesWithRegexpMatch(messages.DecodeError, 'Expected type DateTimeMessage, got VoidMessage: <VoidMessage>', field.value_from_message, message_types.VoidMessage())
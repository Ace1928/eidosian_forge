import datetime
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testValueToMessageBadValue(self):
    field = message_types.DateTimeField(1)
    self.assertRaisesWithRegexpMatch(messages.EncodeError, 'Expected type datetime, got int: 20', field.value_to_message, 20)
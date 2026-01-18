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
def testValidate_Repeated(self):
    """Tests validation of repeated fields."""

    class SimpleMessage(messages.Message):
        repeated = messages.IntegerField(1, repeated=True)
    simple_message = SimpleMessage()
    for valid_value in ([], [10], [10, 20], (), (10,), (10, 20)):
        simple_message.repeated = valid_value
        simple_message.check_initialized()
    simple_message.repeated = []
    simple_message.check_initialized()
    for invalid_value in (10, ['10', '20'], [None], (None,)):
        self.assertRaises(messages.ValidationError, setattr, simple_message, 'repeated', invalid_value)
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
def testValidate_Optional(self):
    """Tests validation of optional fields."""

    class SimpleMessage(messages.Message):
        non_required = messages.IntegerField(1)
    simple_message = SimpleMessage()
    simple_message.check_initialized()
    simple_message.non_required = 10
    simple_message.check_initialized()
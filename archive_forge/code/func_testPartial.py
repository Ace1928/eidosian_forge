import cgi
import datetime
import inspect
import os
import re
import socket
import types
import unittest
import six
from six.moves import range  # pylint: disable=redefined-builtin
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def testPartial(self):
    """Test message with a few values set."""
    message = OptionalMessage()
    message.double_value = 1.23
    message.int64_value = -100000000000
    message.int32_value = 1020
    message.string_value = u'a string'
    message.enum_value = OptionalMessage.SimpleEnum.VAL2
    self.EncodeDecode(self.encoded_partial, message)
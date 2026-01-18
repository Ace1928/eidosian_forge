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
def testDecodeInvalidRepeatedEnumType(self):
    decoded = self.PROTOLIB.decode_message(RepeatedMessage, self.encoded_invalid_repeated_enum)
    message = RepeatedMessage()
    message.enum_value = [RepeatedMessage.SimpleEnum.VAL1]
    self.assertEqual(message, decoded)
    encoded = self.PROTOLIB.encode_message(decoded)
    self.assertEqual(self.encoded_invalid_repeated_enum, encoded)
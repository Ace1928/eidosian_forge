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
def testFull(self):
    """Test all types."""
    message = OptionalMessage()
    message.double_value = 1.23
    message.float_value = -2.5
    message.int64_value = -100000000000
    message.uint64_value = 102020202020
    message.int32_value = 1020
    message.bool_value = True
    message.string_value = u'a stringя'
    message.bytes_value = b'a bytes\xff\xfe'
    message.enum_value = OptionalMessage.SimpleEnum.VAL2
    self.EncodeDecode(self.encoded_full, message)
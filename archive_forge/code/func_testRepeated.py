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
def testRepeated(self):
    """Test repeated fields."""
    message = RepeatedMessage()
    message.double_value = [1.23, 2.3]
    message.float_value = [-2.5, 0.5]
    message.int64_value = [-100000000000, 20]
    message.uint64_value = [102020202020, 10]
    message.int32_value = [1020, 718]
    message.bool_value = [True, False]
    message.string_value = [u'a stringя', u'another string']
    message.bytes_value = [b'a bytes\xff\xfe', b'another bytes']
    message.enum_value = [RepeatedMessage.SimpleEnum.VAL2, RepeatedMessage.SimpleEnum.VAL1]
    self.EncodeDecode(self.encoded_repeated, message)
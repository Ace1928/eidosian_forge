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
def testEncodingRepeatedNestedEmptyMessage(self):
    """Test encoding a nested empty message."""
    message = HasOptionalNestedMessage()
    message.repeated_nested = [OptionalMessage(), OptionalMessage()]
    self.EncodeDecode(self.encoded_repeated_nested_empty, message)
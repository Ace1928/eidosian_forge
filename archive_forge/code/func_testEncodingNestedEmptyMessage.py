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
def testEncodingNestedEmptyMessage(self):
    """Test encoding a nested empty message."""
    message = HasOptionalNestedMessage()
    message.nested = OptionalMessage()
    self.EncodeDecode(self.encoded_nested_empty, message)
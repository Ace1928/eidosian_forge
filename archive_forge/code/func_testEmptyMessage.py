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
def testEmptyMessage(self):
    self.EncodeDecode(self.encoded_empty_message, OptionalMessage())
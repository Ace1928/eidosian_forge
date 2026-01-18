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
def testDocstring(self):
    """Test that docstring is supported ok."""

    class NotImportant(messages.Enum):
        """I have a docstring."""
        VALUE1 = 1
    self.assertEquals('I have a docstring.', NotImportant.__doc__)
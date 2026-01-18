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
def testDefaultFields_None(self):
    """Test none is always acceptable."""

    def action(field_class):
        field_class(1, default=None)
        field_class(1, required=True, default=None)
        field_class(1, repeated=True, default=None)
    self.ActionOnAllFieldClasses(action)
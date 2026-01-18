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
def testNumbers(self):
    """Tests that numbers iterates of enum numbers."""
    self.assertEquals(set([2, 4, 5, 20, 40, 50, 80]), set(Color.numbers()))
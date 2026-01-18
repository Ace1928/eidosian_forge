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
def testNaturalOrder(self):
    """Test that natural order enumeration is in numeric order."""
    self.assertEquals([Color.ORANGE, Color.GREEN, Color.INDIGO, Color.RED, Color.YELLOW, Color.BLUE, Color.VIOLET], sorted(Color))
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
def testNames(self):
    """Test that names iterates over enum names."""
    self.assertEquals(set(['BLUE', 'GREEN', 'INDIGO', 'ORANGE', 'RED', 'VIOLET', 'YELLOW']), set(Color.names()))
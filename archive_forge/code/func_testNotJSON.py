import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testNotJSON(self):
    """Test error when string is not valid JSON."""
    self.assertRaises(ValueError, protojson.decode_message, MyMessage, '{this is not json}')
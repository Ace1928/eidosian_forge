import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testEmptyList(self):
    """Test that empty lists are ignored."""
    self.assertEquals(MyMessage(), protojson.decode_message(MyMessage, '{"a_repeated": []}'))
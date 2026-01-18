import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testConvertStringToNumbers(self):
    """Test that strings passed to integer fields are converted."""
    message = protojson.decode_message(MyMessage, '{"an_integer": "10",\n                                           "a_float": "3.5",\n                                           "a_repeated": ["1", "2"],\n                                           "a_repeated_float": ["1.5", "2", 10]\n                                           }')
    self.assertEquals(MyMessage(an_integer=10, a_float=3.5, a_repeated=[1, 2], a_repeated_float=[1.5, 2.0, 10.0]), message)
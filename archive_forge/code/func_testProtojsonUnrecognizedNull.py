import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testProtojsonUnrecognizedNull(self):
    """Test that unrecognized fields that are None are skipped."""
    decoded = protojson.decode_message(MyMessage, '{"an_integer": 1, "unrecognized_null": null}')
    self.assertEquals(decoded.an_integer, 1)
    self.assertEquals(decoded.all_unrecognized_fields(), [])
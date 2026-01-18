import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testProtojsonUnrecognizedFieldName(self):
    """Test that unrecognized fields are saved and can be accessed."""
    decoded = protojson.decode_message(MyMessage, '{"an_integer": 1, "unknown_val": 2}')
    self.assertEquals(decoded.an_integer, 1)
    self.assertEquals(1, len(decoded.all_unrecognized_fields()))
    self.assertEquals('unknown_val', decoded.all_unrecognized_fields()[0])
    self.assertEquals((2, messages.Variant.INT64), decoded.get_unrecognized_field_info('unknown_val'))
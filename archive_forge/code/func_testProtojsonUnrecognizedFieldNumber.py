import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testProtojsonUnrecognizedFieldNumber(self):
    """Test that unrecognized fields are saved and can be accessed."""
    decoded = protojson.decode_message(MyMessage, '{"an_integer": 1, "1001": "unknown", "-123": "negative", "456_mixed": 2}')
    self.assertEquals(decoded.an_integer, 1)
    self.assertEquals(3, len(decoded.all_unrecognized_fields()))
    self.assertFalse(1001 in decoded.all_unrecognized_fields())
    self.assertTrue('1001' in decoded.all_unrecognized_fields())
    self.assertEquals(('unknown', messages.Variant.STRING), decoded.get_unrecognized_field_info('1001'))
    self.assertTrue('-123' in decoded.all_unrecognized_fields())
    self.assertEquals(('negative', messages.Variant.STRING), decoded.get_unrecognized_field_info('-123'))
    self.assertTrue('456_mixed' in decoded.all_unrecognized_fields())
    self.assertEquals((2, messages.Variant.INT64), decoded.get_unrecognized_field_info('456_mixed'))
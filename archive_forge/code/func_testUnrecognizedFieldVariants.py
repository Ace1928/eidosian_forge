import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testUnrecognizedFieldVariants(self):
    """Test that unrecognized fields are mapped to the right variants."""
    for encoded, expected_variant in (('{"an_integer": 1, "unknown_val": 2}', messages.Variant.INT64), ('{"an_integer": 1, "unknown_val": 2.0}', messages.Variant.DOUBLE), ('{"an_integer": 1, "unknown_val": "string value"}', messages.Variant.STRING), ('{"an_integer": 1, "unknown_val": [1, 2, 3]}', messages.Variant.INT64), ('{"an_integer": 1, "unknown_val": [1, 2.0, 3]}', messages.Variant.DOUBLE), ('{"an_integer": 1, "unknown_val": [1, "foo", 3]}', messages.Variant.STRING), ('{"an_integer": 1, "unknown_val": true}', messages.Variant.BOOL)):
        decoded = protojson.decode_message(MyMessage, encoded)
        self.assertEquals(decoded.an_integer, 1)
        self.assertEquals(1, len(decoded.all_unrecognized_fields()))
        self.assertEquals('unknown_val', decoded.all_unrecognized_fields()[0])
        _, decoded_variant = decoded.get_unrecognized_field_info('unknown_val')
        self.assertEquals(expected_variant, decoded_variant)
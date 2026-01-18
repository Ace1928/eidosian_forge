import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testNullValues(self):
    """Test that null values overwrite existing values."""
    self.assertEquals(MyMessage(), protojson.decode_message(MyMessage, '{"an_integer": null, "a_nested": null, "an_enum": null}'))
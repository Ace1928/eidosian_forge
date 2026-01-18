import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testMergeEmptyString(self):
    """Test merging the empty or space only string."""
    message = protojson.decode_message(test_util.OptionalMessage, '')
    self.assertEquals(test_util.OptionalMessage(), message)
    message = protojson.decode_message(test_util.OptionalMessage, ' ')
    self.assertEquals(test_util.OptionalMessage(), message)
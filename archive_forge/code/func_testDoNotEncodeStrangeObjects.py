import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testDoNotEncodeStrangeObjects(self):
    """Test trying to encode a strange object.

        The main purpose of this test is to complete coverage. It
        ensures that the default behavior of the JSON encoder is
        preserved when someone tries to serialized an unexpected type.

        """

    class BogusObject(object):

        def check_initialized(self):
            pass
    self.assertRaises(TypeError, protojson.encode_message, BogusObject())
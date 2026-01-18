import base64
import datetime
import json
import sys
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testMessageToRepr(self):
    msg = SimpleMessage(field='field', repfield=['field', 'field'])
    self.assertEqual(encoding.MessageToRepr(msg), "%s.SimpleMessage(field='field',repfield=['field','field',],)" % (__name__,))
    self.assertEqual(encoding.MessageToRepr(msg, no_modules=True), "SimpleMessage(field='field',repfield=['field','field',],)")
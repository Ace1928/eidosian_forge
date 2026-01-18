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
def testUnrecognizedFieldIterRepeated(self):
    m = encoding.DictToMessage({'msg_field': [{'field': 'foo'}, {'not_a_field': 'bar'}]}, RepeatedNestedMessage)
    results = list(encoding.UnrecognizedFieldIter(m))
    self.assertEqual(1, len(results))
    edges, fields = results[0]
    expected_edge = encoding.ProtoEdge(encoding.EdgeType.REPEATED, 'msg_field', 1)
    self.assertEqual((expected_edge,), edges)
    self.assertEqual(['not_a_field'], fields)
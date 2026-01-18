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
def testUnrecognizedFieldIterNestedMap(self):
    m = encoding.DictToMessage({'map_field': [{'msg_field': {'foo': {'field_one': 1}, 'bar': {'not_a_field': 1}}}]}, RepeatedNestedMapMessage)
    results = list(encoding.UnrecognizedFieldIter(m))
    self.assertEqual(1, len(results))
    edges, fields = results[0]
    expected_edges = (encoding.ProtoEdge(encoding.EdgeType.REPEATED, 'map_field', 0), encoding.ProtoEdge(encoding.EdgeType.MAP, 'msg_field', 'bar'))
    self.assertEqual(expected_edges, edges)
    self.assertEqual(['not_a_field'], fields)
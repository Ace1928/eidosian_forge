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
def testUnrecognizedFieldIter(self):
    m = encoding.DictToMessage({'nested': {'nested': {'a': 'b'}, 'nested_list': ['foo'], 'extra_field': 'foo'}}, ExtraNestedMessage)
    results = list(encoding.UnrecognizedFieldIter(m))
    self.assertEqual(1, len(results))
    edges, fields = results[0]
    expected_edge = encoding.ProtoEdge(encoding.EdgeType.SCALAR, 'nested', None)
    self.assertEqual((expected_edge,), edges)
    self.assertEqual(['extra_field'], fields)
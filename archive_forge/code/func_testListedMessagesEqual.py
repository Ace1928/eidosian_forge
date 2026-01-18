import unittest
import httplib2
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 import \
from samples.fusiontables_sample.fusiontables_v1 import \
def testListedMessagesEqual(self):
    self.assertTrue(mock._MessagesEqual(_NestedListMessage(nested_list=[_NestedMessage(nested='foo')]), _NestedListMessage(nested_list=[_NestedMessage(nested='foo')])))
    self.assertTrue(mock._MessagesEqual(_NestedListMessage(nested_list=[_NestedMessage(nested='foo'), _NestedMessage(nested='foo2')]), _NestedListMessage(nested_list=[_NestedMessage(nested='foo'), _NestedMessage(nested='foo2')])))
    self.assertFalse(mock._MessagesEqual(_NestedListMessage(nested_list=[_NestedMessage(nested='foo')]), _NestedListMessage(nested_list=[_NestedMessage(nested='bar')])))
    self.assertFalse(mock._MessagesEqual(_NestedListMessage(nested_list=[_NestedMessage(nested='foo')]), _NestedListMessage(nested_list=[_NestedMessage(nested='foo'), _NestedMessage(nested='foo')])))
import unittest
import httplib2
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 import \
from samples.fusiontables_sample.fusiontables_v1 import \
def testClientUnmock(self):
    mock_client = mock.Client(fusiontables.FusiontablesV1)
    self.assertFalse(isinstance(mock_client, fusiontables.FusiontablesV1))
    attributes = set(mock_client.__dict__.keys())
    mock_client = mock_client.Mock()
    self.assertTrue(isinstance(mock_client, fusiontables.FusiontablesV1))
    self.assertTrue(set(mock_client.__dict__.keys()) - attributes)
    mock_client.Unmock()
    self.assertFalse(isinstance(mock_client, fusiontables.FusiontablesV1))
    self.assertEqual(attributes, set(mock_client.__dict__.keys()))
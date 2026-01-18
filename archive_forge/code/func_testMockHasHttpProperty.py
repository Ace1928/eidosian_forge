import unittest
import httplib2
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 import \
from samples.fusiontables_sample.fusiontables_v1 import \
def testMockHasHttpProperty(self):
    with mock.Client(fusiontables.FusiontablesV1) as mock_client:
        self.assertIsInstance(mock_client.http, httplib2.Http)
    self.assertFalse(hasattr(mock_client, 'http'))
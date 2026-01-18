import unittest
import httplib2
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 import \
from samples.fusiontables_sample.fusiontables_v1 import \
def testMockHasOverrideHttpProperty(self):
    real_client = fusiontables.FusiontablesV1(url='http://localhost:8080', http='SomeHttpObject', get_credentials=False)
    with mock.Client(fusiontables.FusiontablesV1, real_client) as mock_client:
        self.assertEquals('SomeHttpObject', mock_client.http)
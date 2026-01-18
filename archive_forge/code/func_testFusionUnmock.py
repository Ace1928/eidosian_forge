import unittest
import httplib2
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 import \
from samples.fusiontables_sample.fusiontables_v1 import \
def testFusionUnmock(self):
    with mock.Client(fusiontables.FusiontablesV1):
        client = fusiontables.FusiontablesV1(get_credentials=False)
        mocked_service_type = type(client.column)
    client = fusiontables.FusiontablesV1(get_credentials=False)
    self.assertNotEqual(type(client.column), mocked_service_type)
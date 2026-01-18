import unittest
import httplib2
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 import \
from samples.fusiontables_sample.fusiontables_v1 import \
def testMockFusionException(self):
    with mock.Client(fusiontables.FusiontablesV1) as client_class:
        client_class.column.List.Expect(request=1, exception=exceptions.HttpError({'status': 404}, '', ''), enable_type_checking=False)
        client = fusiontables.FusiontablesV1(get_credentials=False)
        with self.assertRaises(exceptions.HttpError):
            client.column.List(1)
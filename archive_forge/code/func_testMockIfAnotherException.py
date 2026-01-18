import unittest
import httplib2
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 import \
from samples.fusiontables_sample.fusiontables_v1 import \
def testMockIfAnotherException(self):
    with self.assertRaises(CustomException):
        with mock.Client(fusiontables.FusiontablesV1) as client_class:
            client_class.column.List.Expect(request=1, response=2, enable_type_checking=False)
            raise CustomException('Something when wrong')
import unittest
import httplib2
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 import \
from samples.fusiontables_sample.fusiontables_v1 import \
def testMockFusionTypeCheckingErrors(self):
    with mock.Client(fusiontables.FusiontablesV1) as client_class:
        messages = client_class.MESSAGES_MODULE
        with self.assertRaises(exceptions.ConfigurationValueError):
            client_class.column.List.Expect(messages.FusiontablesColumnInsertRequest(), messages.ColumnList(items=[], totalItems=0))
        with self.assertRaises(exceptions.ConfigurationValueError):
            client_class.column.List.Expect(messages.FusiontablesColumnListRequest(tableId='foo'), messages.Column())
        client_class.column.List.Expect(messages.FusiontablesColumnInsertRequest(), messages.Column(), enable_type_checking=False)
        client_class.column.List(messages.FusiontablesColumnInsertRequest())
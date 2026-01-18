import unittest
import httplib2
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 import \
from samples.fusiontables_sample.fusiontables_v1 import \
def testRequestMacher(self):

    class Matcher(object):

        def __init__(self, eq):
            self._eq = eq

        def __eq__(self, other):
            return self._eq(other)
    with mock.Client(fusiontables.FusiontablesV1) as client_class:

        def IsEven(x):
            return x % 2 == 0

        def IsOdd(x):
            return not IsEven(x)
        client_class.column.List.Expect(request=Matcher(IsEven), response=1, enable_type_checking=False)
        client_class.column.List.Expect(request=Matcher(IsOdd), response=2, enable_type_checking=False)
        client_class.column.List.Expect(request=Matcher(IsEven), response=3, enable_type_checking=False)
        client_class.column.List.Expect(request=Matcher(IsOdd), response=4, enable_type_checking=False)
        client = fusiontables.FusiontablesV1(get_credentials=False)
        self.assertEqual(client.column.List(2), 1)
        self.assertEqual(client.column.List(1), 2)
        self.assertEqual(client.column.List(20), 3)
        self.assertEqual(client.column.List(23), 4)
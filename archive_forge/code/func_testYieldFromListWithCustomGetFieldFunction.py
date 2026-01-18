import unittest
from apitools.base.py import list_pager
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 \
from samples.fusiontables_sample.fusiontables_v1 \
from samples.iam_sample.iam_v1 import iam_v1_client as iam_client
from samples.iam_sample.iam_v1 import iam_v1_messages as iam_messages
def testYieldFromListWithCustomGetFieldFunction(self):
    self.mocked_client.column.List.Expect(messages.FusiontablesColumnListRequest(maxResults=100, pageToken=None, tableId='mytable'), messages.ColumnList(items=[messages.Column(name='c0')]))
    custom_getter_called = []

    def Custom_Getter(message, attribute):
        custom_getter_called.append(True)
        return getattr(message, attribute)
    client = fusiontables.FusiontablesV1(get_credentials=False)
    request = messages.FusiontablesColumnListRequest(tableId='mytable')
    results = list_pager.YieldFromList(client.column, request, get_field_func=Custom_Getter)
    self._AssertInstanceSequence(results, 1)
    self.assertEquals(1, len(custom_getter_called))
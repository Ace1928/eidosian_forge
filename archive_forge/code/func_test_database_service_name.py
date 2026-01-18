from unittest import mock
import fixtures
from keystoneauth1 import adapter
import logging
import requests
import testtools
from troveclient.apiclient import client
from troveclient import client as other_client
from troveclient import exceptions
from troveclient import service_catalog
import troveclient.v1.client
@mock.patch.object(adapter.LegacyJsonAdapter, 'request')
def test_database_service_name(self, m_request):
    m_request.return_value = (mock.MagicMock(status_code=200), None)
    client = other_client.SessionClient(session=mock.MagicMock(), auth=mock.MagicMock())
    client.request('http://no.where', 'GET')
    self.assertIsNone(client.database_service_name)
    client = other_client.SessionClient(session=mock.MagicMock(), auth=mock.MagicMock(), database_service_name='myservice')
    client.request('http://no.where', 'GET')
    self.assertEqual('myservice', client.database_service_name)
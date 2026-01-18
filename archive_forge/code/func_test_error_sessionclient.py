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
@mock.patch.object(adapter.LegacyJsonAdapter, 'get_endpoint', return_value=None)
def test_error_sessionclient(self, m_end_point, m_request):
    m_request.return_value = (mock.MagicMock(status_code=200), None)
    self.assertRaises(exceptions.EndpointNotFound, other_client.SessionClient, session=mock.MagicMock(), auth=mock.MagicMock())
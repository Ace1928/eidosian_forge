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
@mock.patch.object(service_catalog, 'ServiceCatalog')
def test_client_auth_token_v1_auth(self, mock_service_catalog):
    auth_url = 'http://www.blah.com'
    proxy_token = 'foobar'
    mock_service_catalog.return_value.get_token = mock.Mock(return_value=proxy_token)
    instance = other_client.HTTPClient(user='user', password='password', projectid='projectid', timeout=2, auth_url=auth_url)
    instance.management_url = 'http://example.com'
    instance.get_service_url = mock.Mock(return_value='http://example.com')
    instance.version = 'v1.0'
    mock_request = mock.Mock()
    mock_request.return_value = requests.Response()
    mock_request.return_value.status_code = 200
    mock_request.return_value.headers = {'x-server-management-url': 'blah.com'}
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': 'python-troveclient'}
    with mock.patch('requests.request', mock_request):
        instance.authenticate()
        called_args, called_kwargs = mock_request.call_args
        self.assertEqual(('POST', 'http://www.blah.com/v2.0/tokens'), called_args)
        self.assertEqual(headers, called_kwargs['headers'])
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
def test_construct_http_client(self):
    mock_request = mock.Mock()
    mock_request.return_value = requests.Response()
    mock_request.return_value.status_code = 200
    mock_request.return_value.headers = {'x-server-management-url': 'blah.com', 'x-auth-token': 'blah'}
    with mock.patch('requests.request', mock_request):
        self.assertIsInstance(other_client._construct_http_client(), other_client.HTTPClient)
        self.assertIsInstance(other_client._construct_http_client(session=mock.Mock(), auth=mock.Mock()), other_client.SessionClient)
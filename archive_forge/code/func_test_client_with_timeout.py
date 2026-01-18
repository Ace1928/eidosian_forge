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
def test_client_with_timeout(self):
    instance = other_client.HTTPClient(user='user', password='password', projectid='project', timeout=2, auth_url='http://www.blah.com', insecure=True)
    self.assertEqual(2, instance.timeout)
    mock_request = mock.Mock()
    mock_request.return_value = requests.Response()
    mock_request.return_value.status_code = 200
    mock_request.return_value.headers = {'x-server-management-url': 'blah.com', 'x-auth-token': 'blah'}
    with mock.patch('requests.request', mock_request):
        instance.authenticate()
        requests.request.assert_called_with(mock.ANY, mock.ANY, timeout=2, headers=mock.ANY, verify=mock.ANY)
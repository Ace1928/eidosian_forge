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
def test_client_bad_request(self):
    instance = other_client.HTTPClient(user='user', password='password', projectid='project', timeout=2, auth_url='http://www.blah.com')
    instance.auth_token = 'foobar'
    instance.management_url = 'http://example.com'
    instance.get_service_url = mock.Mock(return_value='http://example.com')
    instance.version = 'v2.0'
    mock_request = mock.Mock()
    mock_request.side_effect = other_client.exceptions.BadRequest()
    with mock.patch('requests.request', mock_request):
        self.assertRaises(exceptions.BadRequest, instance.get, '/instances')
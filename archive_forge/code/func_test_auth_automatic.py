import json
from unittest import mock
import requests
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.v3 import client
def test_auth_automatic(self):
    cs = client.Client('username', 'password', 'project_id', 'auth_url')
    http_client = cs.client
    http_client.management_url = ''
    mock_request = mock.Mock(return_value=(None, None))

    @mock.patch.object(http_client, 'request', mock_request)
    @mock.patch.object(http_client, 'authenticate')
    def test_auth_call(m):
        http_client.get('/')
        self.assertTrue(m.called)
        self.assertTrue(mock_request.called)
    test_auth_call()
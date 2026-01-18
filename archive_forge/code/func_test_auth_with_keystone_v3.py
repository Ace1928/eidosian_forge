import json
from unittest import mock
import uuid
import requests
from cinderclient import client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
def test_auth_with_keystone_v3(self):
    cl = get_authed_client()
    cl.auth_url = 'http://example.com:5000/v3'

    @mock.patch.object(requests, 'request', mock_201_request)
    def test_auth_call():
        cl.authenticate()
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': cl.USER_AGENT}
        data = {'auth': {'scope': {'project': {'domain': {'name': 'Default'}, 'name': 'project_id'}}, 'identity': {'methods': ['password'], 'password': {'user': {'domain': {'name': 'Default'}, 'password': 'password', 'name': 'username'}}}}}
        actual_data = mock_201_request.call_args[1]['data']
        self.assertDictEqual(data, json.loads(actual_data))
        mock_201_request.assert_called_with('POST', 'http://example.com:5000/v3/auth/tokens', headers=headers, allow_redirects=True, data=actual_data, **self.TEST_REQUEST_BASE)
    test_auth_call()
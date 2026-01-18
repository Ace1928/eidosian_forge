from unittest import mock
from openstack import config
from ironicclient import client as iroclient
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import client as v1
def test_get_client_with_auth_token_endpoint(self):
    kwargs = {'endpoint': 'http://localhost:6385/v1', 'token': 'USER_AUTH_TOKEN'}
    client = self._test_get_client(auth='admin_token', **kwargs)
    self.assertIsInstance(client.http_client, http.SessionClient)
    self.assertEqual('http://localhost:6385', client.http_client.endpoint_override)
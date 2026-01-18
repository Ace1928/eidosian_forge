from unittest import mock
from openstack import config
from ironicclient import client as iroclient
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import client as v1
def test_get_client_additional_headers_and_global_request(self):
    req_id = 'req-7b081d28-8272-45f4-9cf6-89649c1c7a1a'
    kwargs = {'endpoint': 'http://localhost:6385/v1', 'additional_headers': {'foo': 'bar'}, 'global_request_id': req_id}
    client = self._test_get_client(auth='none', **kwargs)
    self.assertIsInstance(client.http_client, http.SessionClient)
    self.assertEqual('http://localhost:6385', client.http_client.endpoint_override)
    self.assertEqual(req_id, client.http_client.global_request_id)
    self.assertEqual({'foo': 'bar'}, client.http_client.additional_headers)
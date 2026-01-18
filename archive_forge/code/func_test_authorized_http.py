from unittest import mock
import urllib.parse
import fixtures
from oslo_serialization import jsonutils
import requests
from requests_mock.contrib import fixture as rm_fixture
from testtools import matchers
import webob
from keystonemiddleware import s3_token
from keystonemiddleware.tests.unit import utils
def test_authorized_http(self):
    protocol = 'http'
    host = 'fakehost'
    port = 35357
    self.requests_mock.post('%s://%s:%s/v2.0/s3tokens' % (protocol, host, port), status_code=201, json=GOOD_RESPONSE)
    self.middleware = s3_token.filter_factory({'auth_protocol': protocol, 'auth_host': host, 'auth_port': port})(FakeApp())
    req = webob.Request.blank('/v1/AUTH_cfa/c/o')
    req.headers['Authorization'] = 'access:signature'
    req.headers['X-Storage-Token'] = 'token'
    req.get_response(self.middleware)
    self.assertTrue(req.path.startswith('/v1/AUTH_TENANT_ID'))
    self.assertEqual(req.headers['X-Auth-Token'], 'TOKEN_ID')
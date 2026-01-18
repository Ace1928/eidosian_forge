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
def test_authorization_nova_toconnect(self):
    req = webob.Request.blank('/v1/AUTH_swiftint/c/o')
    req.headers['Authorization'] = 'access:FORCED_TENANT_ID:signature'
    req.headers['X-Storage-Token'] = 'token'
    req.get_response(self.middleware)
    path = req.environ['PATH_INFO']
    self.assertTrue(path.startswith('/v1/AUTH_FORCED_TENANT_ID'))
import datetime
import os
import time
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystoneauth1 import session
import oslo_cache
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import pbr.version
import testresources
from testtools import matchers
import webob
import webob.dec
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _base
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import client_fixtures
def test_auth_plugin_with_token(self):
    self.requests_mock.get('%s/v3/auth/tokens' % BASE_URI, text=self.token_response, headers={'X-Subject-Token': uuid.uuid4().hex})
    body = uuid.uuid4().hex
    www_authenticate_uri = 'http://local.test'
    conf = {'delay_auth_decision': 'True', 'www_authenticate_uri': www_authenticate_uri, 'auth_type': 'admin_token', 'endpoint': '%s/v3' % BASE_URI, 'token': FAKE_ADMIN_TOKEN_ID}
    middleware = self.create_simple_middleware(body=body, conf=conf)
    resp = self.call(middleware, headers={'X-Auth-Token': 'non-keystone-token'})
    self.assertEqual(body.encode(), resp.body)
    token_auth = resp.request.environ['keystone.token_auth']
    self.assertFalse(token_auth.has_user_token)
    self.assertIsNone(token_auth.user)
    self.assertFalse(token_auth.has_service_token)
    self.assertIsNone(token_auth.service)
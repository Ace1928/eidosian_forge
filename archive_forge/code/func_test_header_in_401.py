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
def test_header_in_401(self):
    body = uuid.uuid4().hex
    www_authenticate_uri = 'http://local.test'
    conf = {'delay_auth_decision': 'True', 'auth_version': 'v3', 'www_authenticate_uri': www_authenticate_uri}
    middleware = self.create_simple_middleware(status='401 Unauthorized', body=body, conf=conf)
    resp = self.call(middleware, expected_status=401)
    self.assertEqual(body.encode(), resp.body)
    self.assertEqual('Keystone uri="%s"' % www_authenticate_uri, resp.headers['WWW-Authenticate'])
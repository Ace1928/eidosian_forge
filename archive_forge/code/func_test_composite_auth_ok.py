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
def test_composite_auth_ok(self):
    token = self.token_dict['uuid_token_default']
    service_token = self.token_dict['uuid_service_token_default']
    fake_logger = fixtures.FakeLogger(level=logging.DEBUG)
    self.middleware.logger = self.useFixture(fake_logger)
    resp = self.call_middleware(headers={'X-Auth-Token': token, 'X-Service-Token': service_token})
    self.assertEqual(FakeApp.SUCCESS, resp.body)
    expected_env = dict(EXPECTED_V2_DEFAULT_ENV_RESPONSE)
    expected_env.update(EXPECTED_V2_DEFAULT_SERVICE_ENV_RESPONSE)
    self.assertIn('Received request from user: ', fake_logger.output)
    self.assertIn('user_id %(HTTP_X_USER_ID)s, project_id %(HTTP_X_TENANT_ID)s, roles ' % expected_env, fake_logger.output)
    self.assertIn('service: user_id %(HTTP_X_SERVICE_USER_ID)s, project_id %(HTTP_X_SERVICE_PROJECT_ID)s, roles ' % expected_env, fake_logger.output)
    roles = ','.join([expected_env['HTTP_X_SERVICE_ROLES'], expected_env['HTTP_X_ROLES']])
    for r in roles.split(','):
        self.assertIn(r, fake_logger.output)
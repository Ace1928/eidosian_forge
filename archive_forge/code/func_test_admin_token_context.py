import copy
import hashlib
from unittest import mock
import uuid
import fixtures
import http.client
import webtest
from keystone.auth import core as auth_core
from keystone.common import authorization
from keystone.common import context as keystone_context
from keystone.common import provider_api
from keystone.common import tokenless_auth
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_backend_sql
def test_admin_token_context(self):
    self.config_fixture.config(admin_token='ADMIN')
    log_fix = self.useFixture(fixtures.FakeLogger())
    headers = {authorization.AUTH_TOKEN_HEADER: 'ADMIN'}
    req = self._do_middleware_request(headers=headers)
    self.assertTrue(req.environ[auth_context.CONTEXT_ENV]['is_admin'])
    self.assertNotIn('Invalid user token', log_fix.output)
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
def test_ephemeral_no_group_found_fail(self):
    env = {}
    env['SSL_CLIENT_I_DN'] = self.client_issuer
    env['HTTP_X_PROJECT_NAME'] = self.project_name
    env['HTTP_X_PROJECT_DOMAIN_NAME'] = self.domain_name
    env['SSL_CLIENT_USER_NAME'] = self.user['name']
    self.config_fixture.config(group='tokenless_auth', protocol='ephemeral')
    self.protocol_id = 'ephemeral'
    mapping = copy.deepcopy(mapping_fixtures.MAPPING_FOR_EPHEMERAL_USER)
    mapping['rules'][0]['local'][0]['group']['id'] = uuid.uuid4().hex
    self._load_mapping_rules(mapping)
    self._middleware_failure(exception.MappedGroupNotFound, extra_environ=env)
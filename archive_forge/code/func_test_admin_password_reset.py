import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_log import log
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_admin_password_reset(self):
    user_ref = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
    old_password_auth = self.build_authentication_request(user_id=user_ref['id'], password=user_ref['password'])
    r = self.v3_create_token(old_password_auth)
    old_token = r.headers.get('X-Subject-Token')
    old_token_auth = self.build_authentication_request(token=old_token)
    self.v3_create_token(old_token_auth)
    new_password = uuid.uuid4().hex
    self.patch('/users/%s' % user_ref['id'], body={'user': {'password': new_password}})
    self.v3_create_token(old_password_auth, expected_status=http.client.UNAUTHORIZED)
    self.v3_create_token(old_token_auth, expected_status=http.client.NOT_FOUND)
    new_password_auth = self.build_authentication_request(user_id=user_ref['id'], password=new_password)
    self.v3_create_token(new_password_auth)
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
def test_lockout_exempt(self):
    self.config_fixture.config(group='security_compliance', lockout_failure_attempts=1)
    self.user_ref = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
    ignore_opt_name = options.IGNORE_LOCKOUT_ATTEMPT_OPT.option_name
    self.user_ref['options'][ignore_opt_name] = True
    PROVIDERS.identity_api.update_user(self.user_ref['id'], self.user_ref)
    bad_password = uuid.uuid4().hex
    self.token = self.get_request_token(bad_password, http.client.UNAUTHORIZED)
    self.get_request_token(self.user_ref['password'], expected_status=http.client.CREATED)
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
def test_change_password_required_upon_first_use_ignore_user(self):
    self.config_fixture.config(group='security_compliance', change_password_upon_first_use=True)
    reset_password = uuid.uuid4().hex
    self.user_ref['password'] = reset_password
    ignore_opt_name = options.IGNORE_CHANGE_PASSWORD_OPT.option_name
    self.user_ref['options'][ignore_opt_name] = True
    PROVIDERS.identity_api.update_user(self.user_ref['id'], self.user_ref)
    self.token = self.get_request_token(reset_password, http.client.CREATED)
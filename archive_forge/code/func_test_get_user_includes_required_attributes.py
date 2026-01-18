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
def test_get_user_includes_required_attributes(self):
    """Call ``GET /users/{user_id}`` required attributes are included."""
    user = unit.new_user_ref(domain_id=self.domain_id, project_id=self.project_id)
    user = PROVIDERS.identity_api.create_user(user)
    self.assertIn('id', user)
    self.assertIn('name', user)
    self.assertIn('enabled', user)
    self.assertIn('password_expires_at', user)
    r = self.get('/users/%(user_id)s' % {'user_id': user['id']})
    self.assertValidUserResponse(r, user)
from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_check_system_grant_for_user(self):
    user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user_id = PROVIDERS.identity_api.create_user(user_ref)['id']
    role = self._create_role()
    self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_system_grant_for_user, user_id, role['id'])
    PROVIDERS.assignment_api.create_system_grant_for_user(user_id, role['id'])
    PROVIDERS.assignment_api.check_system_grant_for_user(user_id, role['id'])
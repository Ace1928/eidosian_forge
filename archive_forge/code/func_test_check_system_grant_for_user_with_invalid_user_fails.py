from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_check_system_grant_for_user_with_invalid_user_fails(self):
    role = self._create_role()
    self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_system_grant_for_user, uuid.uuid4().hex, role['id'])
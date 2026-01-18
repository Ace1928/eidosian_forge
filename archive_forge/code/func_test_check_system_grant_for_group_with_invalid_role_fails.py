from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_check_system_grant_for_group_with_invalid_role_fails(self):
    group_ref = unit.new_group_ref(CONF.identity.default_domain_id)
    group_id = PROVIDERS.identity_api.create_group(group_ref)['id']
    self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_system_grant_for_group, group_id, uuid.uuid4().hex)
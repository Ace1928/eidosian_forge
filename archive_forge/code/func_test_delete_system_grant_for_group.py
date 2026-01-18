from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_delete_system_grant_for_group(self):
    group_ref = unit.new_group_ref(CONF.identity.default_domain_id)
    group_id = PROVIDERS.identity_api.create_group(group_ref)['id']
    role = self._create_role()
    PROVIDERS.assignment_api.create_system_grant_for_group(group_id, role['id'])
    system_roles = PROVIDERS.assignment_api.list_system_grants_for_group(group_id)
    self.assertEqual(len(system_roles), 1)
    PROVIDERS.assignment_api.delete_system_grant_for_group(group_id, role['id'])
    system_roles = PROVIDERS.assignment_api.list_system_grants_for_group(group_id)
    self.assertEqual(len(system_roles), 0)
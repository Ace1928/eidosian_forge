from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_create_system_grant_for_group(self):
    group_ref = unit.new_group_ref(CONF.identity.default_domain_id)
    group_id = PROVIDERS.identity_api.create_group(group_ref)['id']
    role_ref = self._create_role()
    PROVIDERS.assignment_api.create_system_grant_for_group(group_id, role_ref['id'])
    system_roles = PROVIDERS.assignment_api.list_system_grants_for_group(group_id)
    self.assertEqual(len(system_roles), 1)
    self.assertIsNone(system_roles[0]['domain_id'])
    self.assertEqual(system_roles[0]['id'], role_ref['id'])
    self.assertEqual(system_roles[0]['name'], role_ref['name'])
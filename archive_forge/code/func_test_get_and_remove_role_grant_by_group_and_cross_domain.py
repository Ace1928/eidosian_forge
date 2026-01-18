from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_and_remove_role_grant_by_group_and_cross_domain(self):
    group1_domain1_role = unit.new_role_ref()
    PROVIDERS.role_api.create_role(group1_domain1_role['id'], group1_domain1_role)
    group1_domain2_role = unit.new_role_ref()
    PROVIDERS.role_api.create_role(group1_domain2_role['id'], group1_domain2_role)
    domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
    domain2 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
    group1 = unit.new_group_ref(domain_id=domain1['id'])
    group1 = PROVIDERS.identity_api.create_group(group1)
    roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], domain_id=domain1['id'])
    self.assertEqual(0, len(roles_ref))
    roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], domain_id=domain2['id'])
    self.assertEqual(0, len(roles_ref))
    PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=domain1['id'], role_id=group1_domain1_role['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=domain2['id'], role_id=group1_domain2_role['id'])
    roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], domain_id=domain1['id'])
    self.assertDictEqual(group1_domain1_role, roles_ref[0])
    roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], domain_id=domain2['id'])
    self.assertDictEqual(group1_domain2_role, roles_ref[0])
    PROVIDERS.assignment_api.delete_grant(group_id=group1['id'], domain_id=domain2['id'], role_id=group1_domain2_role['id'])
    roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], domain_id=domain2['id'])
    self.assertEqual(0, len(roles_ref))
    self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, group_id=group1['id'], domain_id=domain2['id'], role_id=group1_domain2_role['id'])
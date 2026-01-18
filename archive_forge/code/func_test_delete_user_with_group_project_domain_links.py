import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_delete_user_with_group_project_domain_links(self):
    role1 = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role1['id'], role1)
    domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
    project1 = unit.new_project_ref(domain_id=domain1['id'])
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    user1 = unit.new_user_ref(domain_id=domain1['id'])
    user1 = PROVIDERS.identity_api.create_user(user1)
    group1 = unit.new_group_ref(domain_id=domain1['id'])
    group1 = PROVIDERS.identity_api.create_group(group1)
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=project1['id'], role_id=role1['id'])
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain1['id'], role_id=role1['id'])
    PROVIDERS.identity_api.add_user_to_group(user_id=user1['id'], group_id=group1['id'])
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], project_id=project1['id'])
    self.assertEqual(1, len(roles_ref))
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain1['id'])
    self.assertEqual(1, len(roles_ref))
    PROVIDERS.identity_api.check_user_in_group(user_id=user1['id'], group_id=group1['id'])
    PROVIDERS.identity_api.delete_user(user1['id'])
    self.assertRaises(exception.NotFound, PROVIDERS.identity_api.check_user_in_group, user1['id'], group1['id'])
from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_role_assignments_group_not_found(self):

    def _group_not_found(value):
        raise exception.GroupNotFound(group_id=value)
    for a in PROVIDERS.assignment_api.list_role_assignments():
        PROVIDERS.assignment_api.delete_grant(**a)
    domain_id = CONF.identity.default_domain_id
    group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain_id))
    user1 = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain_id))
    user2 = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain_id))
    PROVIDERS.identity_api.add_user_to_group(user1['id'], group['id'])
    PROVIDERS.identity_api.add_user_to_group(user2['id'], group['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group['id'], domain_id=domain_id, role_id=default_fixtures.MEMBER_ROLE_ID)
    num_assignments = len(PROVIDERS.assignment_api.list_role_assignments())
    self.assertEqual(1, num_assignments)
    with mock.patch.object(PROVIDERS.identity_api, 'get_group', _group_not_found):
        keystone.assignment.COMPUTED_ASSIGNMENTS_REGION.invalidate()
        assignment_list = PROVIDERS.assignment_api.list_role_assignments(include_names=True)
    self.assertEqual(num_assignments, len(assignment_list))
    for assignment in assignment_list:
        includes_group_assignments = False
        if 'group_name' in assignment:
            includes_group_assignments = True
            self.assertEqual('', assignment['group_name'])
            self.assertEqual('', assignment['group_domain_id'])
            self.assertEqual('', assignment['group_domain_name'])
    self.assertTrue(includes_group_assignments)
    num_effective = len(PROVIDERS.assignment_api.list_role_assignments(effective=True))
    self.assertGreater(num_effective, len(assignment_list))
    with mock.patch.object(PROVIDERS.identity_api, 'list_users_in_group', _group_not_found):
        keystone.assignment.COMPUTED_ASSIGNMENTS_REGION.invalidate()
        assignment_list = PROVIDERS.assignment_api.list_role_assignments(effective=True)
    self.assertGreater(num_effective, len(assignment_list))
    PROVIDERS.assignment_api.delete_grant(group_id=group['id'], domain_id=domain_id, role_id=default_fixtures.MEMBER_ROLE_ID)
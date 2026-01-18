from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_role_assignment_by_user_with_domain_group_roles(self):
    """Test listing assignments by user, with group roles on a domain."""
    test_plan = {'entities': {'domains': [{'users': 3, 'groups': 3}, 1], 'roles': 3}, 'group_memberships': [{'group': 0, 'users': [0, 1]}, {'group': 1, 'users': [0]}], 'assignments': [{'user': 0, 'role': 0, 'domain': 0}, {'group': 0, 'role': 1, 'domain': 0}, {'group': 1, 'role': 2, 'domain': 0}, {'user': 1, 'role': 1, 'domain': 0}, {'group': 2, 'role': 2, 'domain': 0}], 'tests': [{'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'domain': 0, 'indirect': {'group': 0}}, {'user': 0, 'role': 2, 'domain': 0, 'indirect': {'group': 1}}]}, {'params': {'user': 0, 'domain': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'domain': 0, 'indirect': {'group': 0}}, {'user': 0, 'role': 2, 'domain': 0, 'indirect': {'group': 1}}]}, {'params': {'user': 0, 'domain': 1, 'effective': True}, 'results': []}, {'params': {'user': 2, 'domain': 0, 'effective': True}, 'results': []}]}
    self.execute_assignment_plan(test_plan)
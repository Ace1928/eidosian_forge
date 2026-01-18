from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_role_assignment_using_sourced_groups(self):
    """Test listing assignments when restricted by source groups."""
    test_plan = {'entities': {'domains': {'id': CONF.identity.default_domain_id, 'users': 3, 'groups': 3, 'projects': 3}, 'roles': 3}, 'group_memberships': [{'group': 0, 'users': [0, 1]}, {'group': 1, 'users': [0]}], 'assignments': [{'user': 0, 'role': 0, 'project': 0}, {'group': 0, 'role': 1, 'project': 1}, {'group': 1, 'role': 2, 'project': 0}, {'group': 1, 'role': 2, 'project': 1}, {'user': 2, 'role': 1, 'project': 1}, {'group': 2, 'role': 2, 'project': 2}], 'tests': [{'params': {'source_from_group_ids': [0, 1], 'effective': True}, 'results': [{'group': 0, 'role': 1, 'project': 1}, {'group': 1, 'role': 2, 'project': 0}, {'group': 1, 'role': 2, 'project': 1}]}, {'params': {'source_from_group_ids': [0, 1], 'role': 2, 'effective': True}, 'results': [{'group': 1, 'role': 2, 'project': 0}, {'group': 1, 'role': 2, 'project': 1}]}]}
    self.execute_assignment_plan(test_plan)
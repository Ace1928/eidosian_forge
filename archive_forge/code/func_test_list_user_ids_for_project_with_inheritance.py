from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_user_ids_for_project_with_inheritance(self):
    test_plan = {'entities': {'domains': {'id': CONF.identity.default_domain_id, 'users': 4, 'groups': 2, 'projects': {'project': 1}}, 'roles': 4}, 'group_memberships': [{'group': 0, 'users': [1]}, {'group': 1, 'users': [3]}], 'assignments': [{'user': 0, 'role': 0, 'project': 1}, {'group': 0, 'role': 1, 'project': 1}, {'user': 2, 'role': 2, 'project': 0, 'inherited_to_projects': True}, {'group': 1, 'role': 3, 'project': 0, 'inherited_to_projects': True}]}
    test_data = self.execute_assignment_plan(test_plan)
    user_ids = PROVIDERS.assignment_api.list_user_ids_for_project(test_data['projects'][1]['id'])
    self.assertThat(user_ids, matchers.HasLength(4))
    for x in range(0, 4):
        self.assertIn(test_data['users'][x]['id'], user_ids)
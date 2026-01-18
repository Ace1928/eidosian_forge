from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_effective_assignments_for_tree(self):
    """Test we correctly list effective assignments for a tree."""
    test_plan = {'entities': {'domains': {'projects': {'project': [{'project': 2}, {'project': 2}]}, 'users': 1}, 'roles': 4}, 'assignments': [{'user': 0, 'role': 1, 'project': 1, 'inherited_to_projects': True}, {'user': 0, 'role': 2, 'project': 2}, {'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 3, 'project': 4}], 'tests': [{'params': {'project': 1, 'effective': True, 'include_subtree': True}, 'results': [{'user': 0, 'role': 1, 'project': 2, 'indirect': {'project': 1}}, {'user': 0, 'role': 1, 'project': 3, 'indirect': {'project': 1}}, {'user': 0, 'role': 2, 'project': 2}]}]}
    self.execute_assignment_plan(test_plan)
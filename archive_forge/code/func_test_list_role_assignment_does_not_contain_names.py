from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_role_assignment_does_not_contain_names(self):
    """Test names are not included with list role assignments.

        Scenario:
            - names are NOT included by default
            - names are NOT included when include_names=False

        """

    def assert_does_not_contain_names(assignment):
        first_asgmt_prj = assignment[0]
        self.assertNotIn('project_name', first_asgmt_prj)
        self.assertNotIn('project_domain_id', first_asgmt_prj)
        self.assertNotIn('user_name', first_asgmt_prj)
        self.assertNotIn('user_domain_id', first_asgmt_prj)
        self.assertNotIn('role_name', first_asgmt_prj)
        self.assertNotIn('role_domain_id', first_asgmt_prj)
    new_role = unit.new_role_ref()
    new_domain = self._get_domain_fixture()
    new_user = unit.new_user_ref(domain_id=new_domain['id'])
    new_project = unit.new_project_ref(domain_id=new_domain['id'])
    new_role = PROVIDERS.role_api.create_role(new_role['id'], new_role)
    new_user = PROVIDERS.identity_api.create_user(new_user)
    PROVIDERS.resource_api.create_project(new_project['id'], new_project)
    PROVIDERS.assignment_api.create_grant(user_id=new_user['id'], project_id=new_project['id'], role_id=new_role['id'])
    role_assign_without_names = PROVIDERS.assignment_api.list_role_assignments(user_id=new_user['id'], project_id=new_project['id'])
    assert_does_not_contain_names(role_assign_without_names)
    role_assign_without_names = PROVIDERS.assignment_api.list_role_assignments(user_id=new_user['id'], project_id=new_project['id'], include_names=False)
    assert_does_not_contain_names(role_assign_without_names)
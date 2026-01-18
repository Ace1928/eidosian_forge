import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_role_assignments_with_implied_roles(self):
    """Call ``GET /role_assignments`` with implied role grant.

        Test Plan:

        - Create a domain with a user and a project
        - Create 3 roles
        - Role 0 implies role 1 and role 1 implies role 2
        - Assign the top role to the project
        - Issue the URL to check effective roles on project - this
          should return all 3 roles.
        - Check the links of the 3 roles indicate the prior role where
          appropriate

        """
    domain, user, project = self._create_test_domain_user_project()
    self._create_three_roles()
    self._create_implied_role(self.role_list[0], self.role_list[1])
    self._create_implied_role(self.role_list[1], self.role_list[2])
    self._assign_top_role_to_user_on_project(user, project)
    response = self.get(self._build_effective_role_assignments_url(user))
    r = response
    self._assert_all_roles_in_assignment(r, user)
    self._assert_initial_assignment_in_effective(response, user, project)
    self._assert_effective_role_for_implied_has_prior_in_links(response, user, project, 0, 1)
    self._assert_effective_role_for_implied_has_prior_in_links(response, user, project, 1, 2)
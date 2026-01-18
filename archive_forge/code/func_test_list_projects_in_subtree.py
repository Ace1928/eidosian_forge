import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.resource.backends import sql as resource_sql
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import utils as test_utils
def test_list_projects_in_subtree(self):
    projects_hierarchy = self._create_projects_hierarchy(hierarchy_size=3)
    project1 = projects_hierarchy[0]
    project2 = projects_hierarchy[1]
    project3 = projects_hierarchy[2]
    project4 = unit.new_project_ref(domain_id=CONF.identity.default_domain_id, parent_id=project2['id'])
    PROVIDERS.resource_api.create_project(project4['id'], project4)
    subtree = PROVIDERS.resource_api.list_projects_in_subtree(project1['id'])
    self.assertEqual(3, len(subtree))
    self.assertIn(project2, subtree)
    self.assertIn(project3, subtree)
    self.assertIn(project4, subtree)
    subtree = PROVIDERS.resource_api.list_projects_in_subtree(project2['id'])
    self.assertEqual(2, len(subtree))
    self.assertIn(project3, subtree)
    self.assertIn(project4, subtree)
    subtree = PROVIDERS.resource_api.list_projects_in_subtree(project3['id'])
    self.assertEqual(0, len(subtree))
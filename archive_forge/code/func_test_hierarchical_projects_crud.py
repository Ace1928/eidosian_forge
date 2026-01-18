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
def test_hierarchical_projects_crud(self):
    projects_hierarchy = self._create_projects_hierarchy(hierarchy_size=1)
    root_project1 = projects_hierarchy[0]
    projects_hierarchy = self._create_projects_hierarchy()
    root_project2 = projects_hierarchy[0]
    leaf_project = projects_hierarchy[1]
    leaf_project['description'] = 'new description'
    PROVIDERS.resource_api.update_project(leaf_project['id'], leaf_project)
    proj_ref = PROVIDERS.resource_api.get_project(leaf_project['id'])
    self.assertDictEqual(leaf_project, proj_ref)
    leaf_project['parent_id'] = root_project1['id']
    self.assertRaises(exception.ForbiddenNotSecurity, PROVIDERS.resource_api.update_project, leaf_project['id'], leaf_project)
    PROVIDERS.resource_api.delete_project(root_project1['id'])
    self.assertRaises(exception.ProjectNotFound, PROVIDERS.resource_api.get_project, root_project1['id'])
    self.assertRaises(exception.ForbiddenNotSecurity, PROVIDERS.resource_api.delete_project, root_project2['id'])
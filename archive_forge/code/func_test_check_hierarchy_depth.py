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
def test_check_hierarchy_depth(self):
    projects_hierarchy = self._create_projects_hierarchy(CONF.max_project_tree_depth)
    leaf_project = projects_hierarchy[CONF.max_project_tree_depth - 1]
    depth = self._get_hierarchy_depth(leaf_project['id'])
    self.assertEqual(CONF.max_project_tree_depth + 1, depth)
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id, parent_id=leaf_project['id'])
    self.assertRaises(exception.ForbiddenNotSecurity, PROVIDERS.resource_api.create_project, project['id'], project)
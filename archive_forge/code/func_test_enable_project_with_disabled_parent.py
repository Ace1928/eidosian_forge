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
def test_enable_project_with_disabled_parent(self):
    projects_hierarchy = self._create_projects_hierarchy()
    root_project = projects_hierarchy[0]
    leaf_project = projects_hierarchy[1]
    leaf_project['enabled'] = False
    PROVIDERS.resource_api.update_project(leaf_project['id'], leaf_project)
    root_project['enabled'] = False
    PROVIDERS.resource_api.update_project(root_project['id'], root_project)
    leaf_project['enabled'] = True
    self.assertRaises(exception.ForbiddenNotSecurity, PROVIDERS.resource_api.update_project, leaf_project['id'], leaf_project)
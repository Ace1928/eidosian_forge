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
def test_disable_hierarchical_not_leaf_project(self):
    projects_hierarchy = self._create_projects_hierarchy()
    root_project = projects_hierarchy[0]
    root_project['enabled'] = False
    self.assertRaises(exception.ForbiddenNotSecurity, PROVIDERS.resource_api.update_project, root_project['id'], root_project)
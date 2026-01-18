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
def test_cannot_enable_cascade_with_parent_disabled(self):
    projects_hierarchy = self._create_projects_hierarchy(hierarchy_size=3)
    grandparent = projects_hierarchy[0]
    parent = projects_hierarchy[1]
    grandparent['enabled'] = False
    PROVIDERS.resource_api.update_project(grandparent['id'], grandparent, cascade=True)
    subtree = PROVIDERS.resource_api.list_projects_in_subtree(parent['id'])
    self.assertFalse(subtree[0]['enabled'])
    parent['enabled'] = True
    self.assertRaises(exception.ForbiddenNotSecurity, PROVIDERS.resource_api.update_project, parent['id'], parent, cascade=True)
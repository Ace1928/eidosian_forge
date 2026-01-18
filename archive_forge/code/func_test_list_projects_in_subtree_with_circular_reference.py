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
def test_list_projects_in_subtree_with_circular_reference(self):
    project1 = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    project1 = PROVIDERS.resource_api.create_project(project1['id'], project1)
    project2 = unit.new_project_ref(domain_id=CONF.identity.default_domain_id, parent_id=project1['id'])
    PROVIDERS.resource_api.create_project(project2['id'], project2)
    project1['parent_id'] = project2['id']
    PROVIDERS.resource_api.driver.update_project(project1['id'], project1)
    subtree = PROVIDERS.resource_api.list_projects_in_subtree(project1['id'])
    self.assertIsNone(subtree)
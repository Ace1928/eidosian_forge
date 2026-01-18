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
@unit.skip_if_no_multiple_domains_support
@test_utils.wip('waiting for sub projects acting as domains support')
def test_create_project_under_domain_hierarchy(self):
    projects_hierarchy = self._create_projects_hierarchy(is_domain=True)
    parent = projects_hierarchy[1]
    project = unit.new_project_ref(domain_id=parent['id'], parent_id=parent['id'], is_domain=False)
    ref = PROVIDERS.resource_api.create_project(project['id'], project)
    self.assertFalse(ref['is_domain'])
    self.assertEqual(parent['id'], ref['parent_id'])
    self.assertEqual(parent['id'], ref['domain_id'])
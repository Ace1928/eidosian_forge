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
def test_create_domain_under_regular_project_hierarchy_fails(self):
    projects_hierarchy = self._create_projects_hierarchy()
    parent = projects_hierarchy[1]
    project = unit.new_project_ref(domain_id=parent['id'], parent_id=parent['id'], is_domain=True)
    self.assertRaises(exception.ValidationError, PROVIDERS.resource_api.create_project, project['id'], project)
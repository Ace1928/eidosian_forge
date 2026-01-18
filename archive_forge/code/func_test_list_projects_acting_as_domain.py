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
def test_list_projects_acting_as_domain(self):
    initial_domains = PROVIDERS.resource_api.list_domains()
    new_projects_acting_as_domains = []
    for i in range(5):
        project = unit.new_project_ref(is_domain=True)
        project = PROVIDERS.resource_api.create_project(project['id'], project)
        new_projects_acting_as_domains.append(project)
    self._create_projects_hierarchy(hierarchy_size=2)
    projects = PROVIDERS.resource_api.list_projects_acting_as_domain()
    expected_number_projects = len(initial_domains) + len(new_projects_acting_as_domains)
    self.assertEqual(expected_number_projects, len(projects))
    for project in new_projects_acting_as_domains:
        self.assertIn(project, projects)
    for domain in initial_domains:
        self.assertIn(domain['id'], [p['id'] for p in projects])
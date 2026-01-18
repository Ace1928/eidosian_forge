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
def test_delete_projects_from_ids(self):
    """Test the resource backend call delete_projects_from_ids.

        Tests the normal flow of the delete_projects_from_ids backend call,
        that ensures no project on the list exists after it is successfully
        called.
        """
    project1_ref = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    project2_ref = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    projects = (project1_ref, project2_ref)
    for project in projects:
        PROVIDERS.resource_api.create_project(project['id'], project)
    projects_ids = [p['id'] for p in projects]
    PROVIDERS.resource_api.driver.delete_projects_from_ids(projects_ids)
    for project_id in projects_ids:
        self.assertRaises(exception.ProjectNotFound, PROVIDERS.resource_api.driver.get_project, project_id)
    PROVIDERS.resource_api.driver.delete_projects_from_ids([])
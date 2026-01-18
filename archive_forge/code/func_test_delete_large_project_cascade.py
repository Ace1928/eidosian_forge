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
def test_delete_large_project_cascade(self):
    """Try delete a large project with cascade true.

        Tree we will create::

               +-p1-+
               |    |
              p5    p2
               |    |
              p6  +-p3-+
                  |    |
                  p7   p4
        """
    projects_hierarchy = self._create_projects_hierarchy(hierarchy_size=4)
    p1 = projects_hierarchy[0]
    self._create_projects_hierarchy(hierarchy_size=2, parent_project_id=p1['id'])
    p3_id = projects_hierarchy[2]['id']
    self._create_projects_hierarchy(hierarchy_size=1, parent_project_id=p3_id)
    prjs_hierarchy = ([p1] + PROVIDERS.resource_api.list_projects_in_subtree(p1['id']))[::-1]
    for project in prjs_hierarchy:
        project['enabled'] = False
        PROVIDERS.resource_api.update_project(project['id'], project)
    PROVIDERS.resource_api.delete_project(p1['id'], cascade=True)
    for project in prjs_hierarchy:
        self.assertRaises(exception.ProjectNotFound, PROVIDERS.resource_api.get_project, project['id'])
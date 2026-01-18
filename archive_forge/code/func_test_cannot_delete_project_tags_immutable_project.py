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
def test_cannot_delete_project_tags_immutable_project(self):
    project, tags = self._create_project_and_tags(num_of_tags=2)
    update_project = {'options': {ro_opt.IMMUTABLE_OPT.option_name: True}}
    PROVIDERS.resource_api.update_project(project['id'], update_project)
    self.assertRaises(exception.ResourceUpdateForbidden, PROVIDERS.resource_api.delete_project_tag, project['id'], tags[0])
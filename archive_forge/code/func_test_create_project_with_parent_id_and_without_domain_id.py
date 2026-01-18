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
@test_utils.wip('waiting for support for parent_id to imply domain_id')
def test_create_project_with_parent_id_and_without_domain_id(self):
    project = unit.new_project_ref(is_domain=True)
    PROVIDERS.resource_api.create_project(project['id'], project)
    sub_project = unit.new_project_ref(parent_id=project['id'])
    ref = PROVIDERS.resource_api.create_project(sub_project['id'], sub_project)
    self.assertEqual(project['domain_id'], ref['domain_id'])
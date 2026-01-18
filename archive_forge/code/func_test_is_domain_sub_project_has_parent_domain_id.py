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
def test_is_domain_sub_project_has_parent_domain_id(self):
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id, is_domain=True)
    PROVIDERS.resource_api.create_project(project['id'], project)
    sub_project = unit.new_project_ref(domain_id=project['id'], parent_id=project['id'], is_domain=True)
    ref = PROVIDERS.resource_api.create_project(sub_project['id'], sub_project)
    self.assertTrue(ref['is_domain'])
    self.assertEqual(project['id'], ref['parent_id'])
    self.assertEqual(project['id'], ref['domain_id'])
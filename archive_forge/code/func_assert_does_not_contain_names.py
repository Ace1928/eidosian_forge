from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def assert_does_not_contain_names(assignment):
    first_asgmt_prj = assignment[0]
    self.assertNotIn('project_name', first_asgmt_prj)
    self.assertNotIn('project_domain_id', first_asgmt_prj)
    self.assertNotIn('user_name', first_asgmt_prj)
    self.assertNotIn('user_domain_id', first_asgmt_prj)
    self.assertNotIn('role_name', first_asgmt_prj)
    self.assertNotIn('role_domain_id', first_asgmt_prj)
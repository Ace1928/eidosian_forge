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
def assert_get_key_is(value):
    project_ref = PROVIDERS.resource_api.update_project(project['id'], project)
    self.assertIs(project_ref.get(key), value)
    project_ref = PROVIDERS.resource_api.get_project(project['id'])
    self.assertIs(project_ref.get(key), value)
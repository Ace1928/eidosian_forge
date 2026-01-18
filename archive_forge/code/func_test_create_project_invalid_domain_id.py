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
def test_create_project_invalid_domain_id(self):
    project = unit.new_project_ref(domain_id=uuid.uuid4().hex)
    self.assertRaises(exception.DomainNotFound, PROVIDERS.resource_api.create_project, project['id'], project)
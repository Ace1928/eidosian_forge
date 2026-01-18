from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_role_assignments_bad_role(self):
    assignment_list = PROVIDERS.assignment_api.list_role_assignments(role_id=uuid.uuid4().hex)
    self.assertEqual([], assignment_list)
from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_user_project_ids_returns_not_found(self):
    self.assertRaises(exception.UserNotFound, PROVIDERS.assignment_api.list_projects_for_user, uuid.uuid4().hex)
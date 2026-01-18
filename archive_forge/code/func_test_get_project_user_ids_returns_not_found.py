from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_project_user_ids_returns_not_found(self):
    self.assertRaises(exception.ProjectNotFound, PROVIDERS.assignment_api.list_user_ids_for_project, uuid.uuid4().hex)
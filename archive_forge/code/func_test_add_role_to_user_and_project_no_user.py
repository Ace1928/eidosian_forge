from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_add_role_to_user_and_project_no_user(self):
    user_id_not_exist = uuid.uuid4().hex
    PROVIDERS.assignment_api.add_role_to_user_and_project(user_id_not_exist, self.project_bar['id'], self.role_admin['id'])
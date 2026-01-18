from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_delete_user_grant_no_user(self):
    role = unit.new_role_ref()
    role_id = role['id']
    PROVIDERS.role_api.create_role(role_id, role)
    user_id = uuid.uuid4().hex
    PROVIDERS.assignment_api.create_grant(role_id, user_id=user_id, project_id=self.project_bar['id'])
    PROVIDERS.assignment_api.delete_grant(role_id, user_id=user_id, project_id=self.project_bar['id'])
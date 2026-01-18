from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_delete_group_grant_no_group(self):
    role = unit.new_role_ref()
    role_id = role['id']
    PROVIDERS.role_api.create_role(role_id, role)
    group_id = uuid.uuid4().hex
    PROVIDERS.assignment_api.create_grant(role_id, group_id=group_id, project_id=self.project_bar['id'])
    PROVIDERS.assignment_api.delete_grant(role_id, group_id=group_id, project_id=self.project_bar['id'])
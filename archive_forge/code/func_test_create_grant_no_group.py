from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_create_grant_no_group(self):
    PROVIDERS.assignment_api.create_grant(self.role_other['id'], group_id=uuid.uuid4().hex, project_id=self.project_bar['id'])
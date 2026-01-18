import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_delete_user_and_check_role_assignment_fails(self):
    """Call ``DELETE`` on the user and check the role assignment."""
    member_url, user = self._create_new_user_and_assign_role_on_project()
    PROVIDERS.identity_api.delete_user(user['id'])
    self.head(member_url, expected_status=http.client.NOT_FOUND)
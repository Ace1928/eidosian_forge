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
def test_get_role_assignments_by_project_user_and_role(self, **filters):
    self.test_get_role_assignments_by_project_and_user(role_id=self.role_id, **filters)
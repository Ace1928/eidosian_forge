import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_delete_user_returns_not_found(self):
    self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.delete_user, uuid.uuid4().hex)
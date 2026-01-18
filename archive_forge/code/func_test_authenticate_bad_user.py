import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_authenticate_bad_user(self):
    with self.make_request():
        self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=uuid.uuid4().hex, password=self.user_foo['password'])
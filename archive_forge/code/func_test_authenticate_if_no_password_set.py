import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_authenticate_if_no_password_set(self):
    id_ = uuid.uuid4().hex
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    PROVIDERS.identity_api.create_user(user)
    with self.make_request():
        self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=id_, password='password')
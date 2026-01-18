import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_create_user_none_password(self):
    user = unit.new_user_ref(password=None, domain_id=CONF.identity.default_domain_id)
    user = PROVIDERS.identity_api.create_user(user)
    PROVIDERS.identity_api.get_user(user['id'])
    with self.make_request():
        self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=user['id'], password='')
        self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=user['id'], password=None)
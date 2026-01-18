import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_create_user_with_long_password(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id, password='a' * 2000)
    PROVIDERS.identity_api.create_user(user)
import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_create_user_doesnt_modify_passed_in_dict(self):
    new_user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    original_user = new_user.copy()
    PROVIDERS.identity_api.create_user(new_user)
    self.assertDictEqual(original_user, new_user)
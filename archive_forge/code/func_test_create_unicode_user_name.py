import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_create_unicode_user_name(self):
    unicode_name = u'name 名字'
    user = unit.new_user_ref(name=unicode_name, domain_id=CONF.identity.default_domain_id)
    ref = PROVIDERS.identity_api.create_user(user)
    self.assertEqual(unicode_name, ref['name'])
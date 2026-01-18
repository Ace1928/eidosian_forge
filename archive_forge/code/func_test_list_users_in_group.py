import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_list_users_in_group(self):
    domain = self._get_domain_fixture()
    new_group = unit.new_group_ref(domain_id=domain['id'])
    new_group = PROVIDERS.identity_api.create_group(new_group)
    user_refs = PROVIDERS.identity_api.list_users_in_group(new_group['id'])
    self.assertEqual([], user_refs)
    new_user = unit.new_user_ref(domain_id=domain['id'])
    new_user = PROVIDERS.identity_api.create_user(new_user)
    PROVIDERS.identity_api.add_user_to_group(new_user['id'], new_group['id'])
    user_refs = PROVIDERS.identity_api.list_users_in_group(new_group['id'])
    found = False
    for x in user_refs:
        if x['id'] == new_user['id']:
            found = True
        self.assertNotIn('password', x)
    self.assertTrue(found)
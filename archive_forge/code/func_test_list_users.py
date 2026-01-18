import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_list_users(self):
    users = PROVIDERS.identity_api.list_users(domain_scope=self._set_domain_scope(CONF.identity.default_domain_id))
    self.assertEqual(len(default_fixtures.USERS), len(users))
    user_ids = set((user['id'] for user in users))
    expected_user_ids = set((getattr(self, 'user_%s' % user['name'])['id'] for user in default_fixtures.USERS))
    for user_ref in users:
        self.assertNotIn('password', user_ref)
    self.assertEqual(expected_user_ids, user_ids)
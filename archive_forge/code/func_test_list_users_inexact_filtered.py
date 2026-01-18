import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_list_users_inexact_filtered(self):
    user_name_data = {5: 'The', 6: 'The Ministry', 7: 'The Ministry of', 8: 'The Ministry of Silly', 9: 'The Ministry of Silly Walks', 10: 'The ministry of silly walks OF'}
    user_list = self._create_test_data('user', 20, domain_id=CONF.identity.default_domain_id, name_dict=user_name_data)
    hints = driver_hints.Hints()
    hints.add_filter('name', 'ministry', comparator='contains')
    users = PROVIDERS.identity_api.list_users(hints=hints)
    self.assertEqual(5, len(users))
    self._match_with_list(users, user_list, list_start=6, list_end=11)
    hints = driver_hints.Hints()
    hints.add_filter('name', 'The', comparator='startswith')
    users = PROVIDERS.identity_api.list_users(hints=hints)
    self.assertEqual(6, len(users))
    self._match_with_list(users, user_list, list_start=5, list_end=11)
    hints = driver_hints.Hints()
    hints.add_filter('name', 'of', comparator='endswith')
    users = PROVIDERS.identity_api.list_users(hints=hints)
    self.assertEqual(2, len(users))
    self.assertIn(user_list[7]['id'], [users[0]['id'], users[1]['id']])
    self.assertIn(user_list[10]['id'], [users[0]['id'], users[1]['id']])
    self._delete_test_data('user', user_list)
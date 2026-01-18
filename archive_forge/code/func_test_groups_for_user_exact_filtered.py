import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_groups_for_user_exact_filtered(self):
    """Test exact filters doesn't break groups_for_user listing."""
    group_list, user_list = self._groups_for_user_data()
    hints = driver_hints.Hints()
    hints.add_filter('name', 'The Ministry', comparator='equals')
    groups = PROVIDERS.identity_api.list_groups_for_user(user_list[0]['id'], hints=hints)
    self.assertEqual(1, len(groups))
    self.assertEqual(group_list[6]['id'], groups[0]['id'])
    self._delete_test_data('user', user_list)
    self._delete_test_data('group', group_list)
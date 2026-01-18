import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_filter_value_wider_than_field(self):
    user_name_field_size = self._get_user_name_field_size()
    if user_name_field_size is None:
        return
    self._create_test_data('user', 2)
    hints = driver_hints.Hints()
    value = 'A' * (user_name_field_size + 1)
    hints.add_filter('name', value)
    users = PROVIDERS.identity_api.list_users(hints=hints)
    self.assertEqual([], users)
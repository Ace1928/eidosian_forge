import uuid
from keystone.common import driver_hints
from keystone import exception
def test_list_groups_for_user_no_groups(self):
    user = self.create_user()
    groups = self.driver.list_groups_for_user(user['id'], driver_hints.Hints())
    self.assertEqual([], groups)
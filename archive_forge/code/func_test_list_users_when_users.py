import uuid
from keystone.common import driver_hints
from keystone import exception
def test_list_users_when_users(self):
    user = self.create_user()
    hints = driver_hints.Hints()
    users = self.driver.list_users(hints)
    self.assertEqual([user['id']], [u['id'] for u in users])
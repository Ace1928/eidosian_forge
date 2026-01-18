import uuid
from keystone.common import driver_hints
from keystone import exception
def test_list_users_no_users(self):
    hints = driver_hints.Hints()
    self.assertEqual([], self.driver.list_users(hints))
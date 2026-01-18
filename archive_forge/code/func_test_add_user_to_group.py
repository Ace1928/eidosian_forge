import uuid
from keystone.common import driver_hints
from keystone import exception
def test_add_user_to_group(self):
    user = self.create_user()
    group = self.create_group()
    self.driver.add_user_to_group(user['id'], group['id'])
    self.driver.check_user_in_group(user['id'], group['id'])
import uuid
from keystone.common import driver_hints
from keystone import exception
def test_list_groups_for_user_no_user(self):
    user_id = uuid.uuid4().hex
    self.assertRaises(exception.UserNotFound, self.driver.list_groups_for_user, user_id, driver_hints.Hints())
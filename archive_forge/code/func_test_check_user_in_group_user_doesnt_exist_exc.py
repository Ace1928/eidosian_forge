import uuid
from keystone.common import driver_hints
from keystone import exception
def test_check_user_in_group_user_doesnt_exist_exc(self):
    group = self.create_group()
    user_id = uuid.uuid4().hex
    self.assertRaises(exception.UserNotFound, self.driver.check_user_in_group, user_id, group['id'])
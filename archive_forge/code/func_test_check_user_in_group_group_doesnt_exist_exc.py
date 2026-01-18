import uuid
from keystone.common import driver_hints
from keystone import exception
def test_check_user_in_group_group_doesnt_exist_exc(self):
    user = self.create_user()
    group_id = uuid.uuid4().hex
    self.assertRaises(exception.GroupNotFound, self.driver.check_user_in_group, user['id'], group_id)
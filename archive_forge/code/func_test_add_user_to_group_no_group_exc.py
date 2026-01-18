import uuid
from keystone.common import driver_hints
from keystone import exception
def test_add_user_to_group_no_group_exc(self):
    user = self.create_user()
    group_id = uuid.uuid4().hex
    self.assertRaises(exception.GroupNotFound, self.driver.add_user_to_group, user['id'], group_id)
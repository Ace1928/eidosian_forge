import uuid
from keystone.common import driver_hints
from keystone import exception
def test_get_group_no_group_exc(self):
    self.assertRaises(exception.GroupNotFound, self.driver.get_group, group_id=uuid.uuid4().hex)
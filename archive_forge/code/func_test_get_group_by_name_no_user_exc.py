import uuid
from keystone.common import driver_hints
from keystone import exception
def test_get_group_by_name_no_user_exc(self):
    self.assertRaises(exception.GroupNotFound, self.driver.get_group_by_name, group_name=uuid.uuid4().hex, domain_id=uuid.uuid4().hex)
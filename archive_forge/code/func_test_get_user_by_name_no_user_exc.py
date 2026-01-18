import uuid
from keystone.common import driver_hints
from keystone import exception
def test_get_user_by_name_no_user_exc(self):
    self.assertRaises(exception.UserNotFound, self.driver.get_user_by_name, user_name=uuid.uuid4().hex, domain_id=uuid.uuid4().hex)
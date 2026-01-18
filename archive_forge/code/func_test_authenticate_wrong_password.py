import uuid
from keystone.common import driver_hints
from keystone import exception
def test_authenticate_wrong_password(self):
    user = self.create_user(password=uuid.uuid4().hex)
    password = uuid.uuid4().hex
    self.assertRaises(AssertionError, self.driver.authenticate, user['id'], password)
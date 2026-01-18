import uuid
from keystone.common import driver_hints
from keystone import exception
def test_create_user_same_name_and_domain_exc(self):
    user1_id = uuid.uuid4().hex
    name = uuid.uuid4().hex
    domain_id = uuid.uuid4().hex
    user = {'id': user1_id, 'name': name, 'enabled': True}
    if self.driver.is_domain_aware():
        user['domain_id'] = domain_id
    self.driver.create_user(user1_id, user)
    user2_id = uuid.uuid4().hex
    user = {'id': user2_id, 'name': name, 'enabled': True}
    if self.driver.is_domain_aware():
        user['domain_id'] = domain_id
    self.assertRaises(exception.Conflict, self.driver.create_user, user2_id, user)
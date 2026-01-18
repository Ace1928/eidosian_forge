import uuid
from keystone.common import driver_hints
from keystone import exception
def test_create_user_all_attributes(self):
    user_id = uuid.uuid4().hex
    user = {'id': user_id, 'name': uuid.uuid4().hex, 'password': uuid.uuid4().hex, 'enabled': True, 'default_project_id': uuid.uuid4().hex, 'password_expires_at': None, 'options': {}}
    if self.driver.is_domain_aware():
        user['domain_id'] = uuid.uuid4().hex
    ret = self.driver.create_user(user_id, user)
    exp_user = user.copy()
    del exp_user['password']
    self.assertEqual(exp_user, ret)
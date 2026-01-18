import datetime
import uuid
import freezegun
import passlib.hash
from keystone.common import password_hashing
from keystone.common import provider_api
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends import base
from keystone.identity.backends import resource_options as iro
from keystone.identity.backends import sql_model as model
from keystone.tests.unit import test_backend_sql
def test_admin_password_reset_is_not_validated_by_password_history(self):
    passwords = [uuid.uuid4().hex, uuid.uuid4().hex]
    user = self._create_user(passwords[0])
    user['password'] = passwords[1]
    with self.make_request():
        PROVIDERS.identity_api.update_user(user['id'], user)
        PROVIDERS.identity_api.authenticate(user_id=user['id'], password=passwords[1])
        user['password'] = passwords[1]
        PROVIDERS.identity_api.update_user(user['id'], user)
        PROVIDERS.identity_api.authenticate(user_id=user['id'], password=passwords[1])
        user['password'] = passwords[0]
        PROVIDERS.identity_api.update_user(user['id'], user)
        PROVIDERS.identity_api.authenticate(user_id=user['id'], password=passwords[0])
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
def assertValidChangePassword(self, user_id, password, new_password):
    with self.make_request():
        PROVIDERS.identity_api.change_password(user_id=user_id, original_password=password, new_password=new_password)
        PROVIDERS.identity_api.authenticate(user_id=user_id, password=new_password)
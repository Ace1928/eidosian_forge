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
def test_authenticate_with_non_expired_password(self):
    password_created_at = datetime.datetime.utcnow() - datetime.timedelta(days=CONF.security_compliance.password_expires_days - 1)
    user = self._create_user(self.user_dict, password_created_at)
    with self.make_request():
        PROVIDERS.identity_api.authenticate(user_id=user['id'], password=self.password)
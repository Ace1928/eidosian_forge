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
def test_authenticate_user_not_disabled_due_to_inactivity(self):
    last_active_at = (datetime.datetime.utcnow() - datetime.timedelta(days=self.max_inactive_days - 1)).date()
    user = self._create_user(self.user_dict, last_active_at)
    with self.make_request():
        user = PROVIDERS.identity_api.authenticate(user_id=user['id'], password=self.password)
    self.assertTrue(user['enabled'])
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
def test_get_user_not_disabled_due_to_inactivity(self):
    user = PROVIDERS.identity_api.create_user(self.user_dict)
    self.assertTrue(user['enabled'])
    last_active_at = (datetime.datetime.utcnow() - datetime.timedelta(self.max_inactive_days - 1)).date()
    self._update_user_last_active_at(user['id'], last_active_at)
    user = PROVIDERS.identity_api.get_user(user['id'])
    self.assertTrue(user['enabled'])
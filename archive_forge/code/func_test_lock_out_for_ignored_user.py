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
def test_lock_out_for_ignored_user(self):
    self.user['options'][iro.IGNORE_LOCKOUT_ATTEMPT_OPT.option_name] = True
    PROVIDERS.identity_api.update_user(self.user['id'], self.user)
    self._fail_auth_repeatedly(self.user['id'])
    with self.make_request():
        self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=self.user['id'], password=uuid.uuid4().hex)
        PROVIDERS.identity_api.authenticate(user_id=self.user['id'], password=self.password)
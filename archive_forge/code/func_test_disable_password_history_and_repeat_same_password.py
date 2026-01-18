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
def test_disable_password_history_and_repeat_same_password(self):
    self.config_fixture.config(group='security_compliance', unique_last_password_count=0)
    password = uuid.uuid4().hex
    user = self._create_user(password)
    self.assertValidChangePassword(user['id'], password, password)
    self.assertValidChangePassword(user['id'], password, password)
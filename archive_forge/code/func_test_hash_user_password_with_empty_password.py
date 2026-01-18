import datetime
import fixtures
import uuid
import freezegun
from oslo_config import fixture as config_fixture
from oslo_log import log
from keystone.common import fernet_utils
from keystone.common import utils as common_utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.server.flask import application
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import utils
def test_hash_user_password_with_empty_password(self):
    password = ''
    user = self._create_test_user(password=password)
    user_hashed = common_utils.hash_user_password(user)
    password_hashed = user_hashed['password']
    self.assertTrue(common_utils.check_password(password, password_hashed))
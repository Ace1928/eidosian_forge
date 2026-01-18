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
def test_hash_long_password_strict(self):
    self.config_fixture.config(strict_password_check=True)
    invalid_length_password = '0' * 9999999
    self.assertRaises(exception.PasswordVerificationError, common_utils.hash_password, invalid_length_password)
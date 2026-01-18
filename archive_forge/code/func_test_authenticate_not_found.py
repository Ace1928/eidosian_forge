import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def test_authenticate_not_found(self):
    self.assertRaises(AssertionError, self.app_cred_api.authenticate, uuid.uuid4().hex, uuid.uuid4().hex)
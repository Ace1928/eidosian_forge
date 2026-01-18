import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def test_authenticate_bad_secret(self):
    app_cred = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'])
    resp = self.app_cred_api.create_application_credential(app_cred)
    badpass = 'badpass'
    self.assertNotEqual(badpass, resp['secret'])
    self.assertRaises(AssertionError, self.app_cred_api.authenticate, resp['id'], badpass)
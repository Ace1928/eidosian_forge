import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def test_application_credential_limits(self):
    config_fixture_ = self.user = self.useFixture(config_fixture.Config())
    config_fixture_.config(group='application_credential', user_limit=2)
    app_cred = self._new_app_cred_data(self.user_foo['id'], self.project_bar['id'])
    self.app_cred_api.create_application_credential(app_cred)
    app_cred['name'] = 'two'
    self.app_cred_api.create_application_credential(app_cred)
    app_cred['name'] = 'three'
    self.assertRaises(exception.ApplicationCredentialLimitExceeded, self.app_cred_api.create_application_credential, app_cred)
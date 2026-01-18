import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def test_deleting_a_user_deletes_application_credentials(self):
    app_cred_1 = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'], name='app1')
    app_cred_2 = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'], name='app2')
    self.app_cred_api.create_application_credential(app_cred_1)
    self.app_cred_api.create_application_credential(app_cred_2)
    self.assertIn(app_cred_1['id'], self._list_ids(self.user_foo))
    self.assertIn(app_cred_2['id'], self._list_ids(self.user_foo))
    self.app_cred_api.get_application_credential(app_cred_1['id'])
    self.app_cred_api.get_application_credential(app_cred_2['id'])
    PROVIDERS.identity_api.delete_user(self.user_foo['id'])
    hints = driver_hints.Hints()
    self.assertListEqual([], self.app_cred_api.list_application_credentials(self.user_foo['id'], hints))
    self.assertRaises(exception.ApplicationCredentialNotFound, self.app_cred_api.get_application_credential, app_cred_1['id'])
    self.assertRaises(exception.ApplicationCredentialNotFound, self.app_cred_api.get_application_credential, app_cred_2['id'])
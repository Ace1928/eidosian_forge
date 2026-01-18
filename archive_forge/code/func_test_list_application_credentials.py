import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def test_list_application_credentials(self):
    app_cred_1 = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'], name='app1')
    app_cred_2 = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'], name='app2')
    app_cred_3 = self._new_app_cred_data(self.user_two['id'], project_id=self.project_baz['id'], name='app3')
    resp1 = self.app_cred_api.create_application_credential(app_cred_1)
    resp2 = self.app_cred_api.create_application_credential(app_cred_2)
    resp3 = self.app_cred_api.create_application_credential(app_cred_3)
    hints = driver_hints.Hints()
    resp = self.app_cred_api.list_application_credentials(self.user_foo['id'], hints)
    resp_ids = [ac['id'] for ac in resp]
    self.assertIn(resp1['id'], resp_ids)
    self.assertIn(resp2['id'], resp_ids)
    self.assertNotIn(resp3['id'], resp_ids)
    for ac in resp:
        self.assertNotIn('secret_hash', ac)
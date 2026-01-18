import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def test_create_application_credential_with_preexisting_access_rules(self):
    app_cred_1 = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'])
    app_cred_1['access_rules'] = [{'id': uuid.uuid4().hex, 'service': uuid.uuid4().hex, 'path': uuid.uuid4().hex, 'method': uuid.uuid4().hex[16:]}]
    resp = self.app_cred_api.create_application_credential(app_cred_1)
    resp_access_rules_1 = resp.pop('access_rules')
    app_cred_2 = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'])
    app_cred_2['access_rules'] = [{'id': resp_access_rules_1[0]['id']}]
    resp = self.app_cred_api.create_application_credential(app_cred_2)
    resp_access_rules_2 = resp.pop('access_rules')
    self.assertDictEqual(resp_access_rules_1[0], resp_access_rules_2[0])
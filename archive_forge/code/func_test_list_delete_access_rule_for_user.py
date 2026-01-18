import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def test_list_delete_access_rule_for_user(self):
    app_cred = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'])
    access_rule_id = uuid.uuid4().hex
    app_cred['access_rules'] = [{'id': access_rule_id, 'service': uuid.uuid4().hex, 'path': uuid.uuid4().hex, 'method': uuid.uuid4().hex[16:]}]
    self.app_cred_api.create_application_credential(app_cred)
    self.assertEqual(1, len(self.app_cred_api.list_access_rules_for_user(self.user_foo['id'])))
    self.app_cred_api.delete_application_credential(app_cred['id'])
    self.assertEqual(1, len(self.app_cred_api.list_access_rules_for_user(self.user_foo['id'])))
    self.app_cred_api.delete_access_rules_for_user(self.user_foo['id'])
    self.assertEqual(0, len(self.app_cred_api.list_access_rules_for_user(self.user_foo['id'])))
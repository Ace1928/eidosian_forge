from openstack import exceptions
from openstack.tests.functional import base
def test_list_access_rules(self):
    app_cred = self._create_application_credential_with_access_rule()
    access_rule_id = app_cred['access_rules'][0]['id']
    access_rules = self.conn.identity.access_rules(user=self.user_id)
    self.assertEqual(1, len(list(access_rules)))
    for access_rule in access_rules:
        self.assertEqual(app_cred['user_id'], self.user_id)
        self.assertEqual(access_rule_id, access_rule['id'])
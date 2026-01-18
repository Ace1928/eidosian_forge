from openstack import exceptions
from openstack.tests.functional import base
def test_find_application_credential(self):
    app_creds = self._create_application_credentials()
    app_cred = self.conn.identity.find_application_credential(user=self.user_id, name_or_id=app_creds['id'])
    self.assertEqual(app_cred['id'], app_creds['id'])
    self.assertEqual(app_cred['user_id'], self.user_id)
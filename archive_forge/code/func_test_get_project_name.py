from castellan.common.credentials import keystone_password
from castellan.tests import base
def test_get_project_name(self):
    self.assertEqual(self.project_name, self.ks_password_credential.project_name)
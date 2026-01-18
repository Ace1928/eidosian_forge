from castellan.common.credentials import keystone_password
from castellan.tests import base
def test_get_project_domain_id(self):
    self.assertEqual(self.project_domain_id, self.ks_password_credential.project_domain_id)
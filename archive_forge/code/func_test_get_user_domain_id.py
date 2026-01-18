from castellan.common.credentials import keystone_password
from castellan.tests import base
def test_get_user_domain_id(self):
    self.assertEqual(self.user_domain_id, self.ks_password_credential.user_domain_id)
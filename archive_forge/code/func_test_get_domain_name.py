from castellan.common.credentials import keystone_password
from castellan.tests import base
def test_get_domain_name(self):
    self.assertEqual(self.domain_name, self.ks_password_credential.domain_name)
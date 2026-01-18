from castellan.common.credentials import keystone_password
from castellan.tests import base
def test_get_domain_id(self):
    self.assertEqual(self.domain_id, self.ks_password_credential.domain_id)
from castellan.common.credentials import keystone_password
from castellan.tests import base
def test_get_trust_id(self):
    self.assertEqual(self.trust_id, self.ks_password_credential.trust_id)
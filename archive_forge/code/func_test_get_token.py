from castellan.common.credentials import token
from castellan.tests import base
def test_get_token(self):
    self.assertEqual(self.token, self.token_credential.token)
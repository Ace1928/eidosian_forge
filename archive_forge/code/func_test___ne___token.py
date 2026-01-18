from castellan.common.credentials import token
from castellan.tests import base
def test___ne___token(self):
    other_token = 'fe32af1fe47e4744a48254e60ae80012'
    other_token_credential = token.Token(other_token)
    self.assertTrue(self.token_credential != other_token_credential)
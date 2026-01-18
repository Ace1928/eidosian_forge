import requests
import testtools.matchers
from keystone.tests.functional import core as functests
def test_get_user_token(self):
    token = self.get_scoped_user_token()
    self.assertIsNotNone(token)
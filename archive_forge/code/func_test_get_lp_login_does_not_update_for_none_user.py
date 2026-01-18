from ... import config
from ...tests import TestCaseInTempDir, TestCaseWithMemoryTransport
from . import account
def test_get_lp_login_does_not_update_for_none_user(self):
    account.get_lp_login()
    self.assertIs(None, account._get_auth_user())
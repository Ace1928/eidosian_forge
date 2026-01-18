from ... import config
from ...tests import TestCaseInTempDir, TestCaseWithMemoryTransport
from . import account
def test_check_lp_login_no_user(self):
    transport = self.get_transport()
    self.assertRaises(account.UnknownLaunchpadUsername, account.check_lp_login, 'test-user', transport)
from ... import config
from ...tests import TestCaseInTempDir, TestCaseWithMemoryTransport
from . import account
def test_get_lp_login(self):
    my_config = config.MemoryStack(b'[DEFAULT]\nlaunchpad_username=test-user\n')
    self.assertEqual('test-user', account.get_lp_login(my_config))
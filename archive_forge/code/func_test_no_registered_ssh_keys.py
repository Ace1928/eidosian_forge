from ... import config
from ...tests import TestCaseInTempDir, TestCaseWithMemoryTransport
from . import account
def test_no_registered_ssh_keys(self):
    error = account.NoRegisteredSSHKeys(user='test-user')
    self.assertEqualDiff('The user test-user has not registered any SSH keys with Launchpad.\nSee <https://launchpad.net/people/+me>', str(error))
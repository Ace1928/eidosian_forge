from ...tests import TestCaseWithTransport
from . import account
def test_login_with_name_verbose(self):
    out, err = self.run_bzr(['launchpad-login', '-v', '--no-check', 'foo'])
    self.assertEqual("Launchpad user ID set to 'foo'.\n", out)
    self.assertEqual('', err)
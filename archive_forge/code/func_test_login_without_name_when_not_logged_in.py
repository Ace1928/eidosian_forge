from ...tests import TestCaseWithTransport
from . import account
def test_login_without_name_when_not_logged_in(self):
    out, err = self.run_bzr(['launchpad-login', '--no-check'], retcode=1)
    self.assertEqual('No Launchpad user ID configured.\n', out)
    self.assertEqual('', err)
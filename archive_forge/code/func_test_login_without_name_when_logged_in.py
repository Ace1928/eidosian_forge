from ...tests import TestCaseWithTransport
from . import account
def test_login_without_name_when_logged_in(self):
    account.set_lp_login('foo')
    out, err = self.run_bzr(['launchpad-login', '--no-check'])
    self.assertEqual('foo\n', out)
    self.assertEqual('', err)
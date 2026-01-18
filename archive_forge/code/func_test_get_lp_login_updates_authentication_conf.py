from ... import config
from ...tests import TestCaseInTempDir, TestCaseWithMemoryTransport
from . import account
def test_get_lp_login_updates_authentication_conf(self):
    account._set_global_option('foo')
    self.assertIs(None, account._get_auth_user())
    account.get_lp_login()
    auth = config.AuthenticationConfig()
    self.assertEqual('foo', account._get_auth_user(auth))
    self.assertEqual('foo', auth.get_user('ssh', 'bazaar.launchpad.net'))
    self.assertEqual('foo', auth.get_user('ssh', 'bazaar.staging.launchpad.net'))
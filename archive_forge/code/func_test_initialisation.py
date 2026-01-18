from twisted.cred.credentials import (
from twisted.cred.test.test_cred import _uhpVersion
from twisted.trial.unittest import TestCase
def test_initialisation(self) -> None:
    """
        The initialisation of L{UsernameHashedPassword} will set C{username}
        and C{hashed} on it.
        """
    UsernameHashedPassword = self.getDeprecatedModuleAttribute('twisted.cred.credentials', 'UsernameHashedPassword', _uhpVersion)
    creds = UsernameHashedPassword(b'foo', b'bar')
    self.assertEqual(creds.username, b'foo')
    self.assertEqual(creds.hashed, b'bar')
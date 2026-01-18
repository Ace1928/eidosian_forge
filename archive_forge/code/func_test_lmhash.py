import warnings
from passlib.tests.utils import TestCase
from passlib.utils.compat import u
def test_lmhash(self):
    from passlib.win32 import raw_lmhash
    for secret, hash in [('OLDPASSWORD', u('c9b81d939d6fd80cd408e6b105741864')), ('NEWPASSWORD', u('09eeab5aa415d6e4d408e6b105741864')), ('welcome', u('c23413a8a1e7665faad3b435b51404ee'))]:
        result = raw_lmhash(secret, hex=True)
        self.assertEqual(result, hash)
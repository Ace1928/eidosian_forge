import hashlib
from binascii import hexlify
from hmac import HMAC
from twisted.cred.credentials import CramMD5Credentials, IUsernameHashedPassword
from twisted.trial.unittest import TestCase
def test_idempotentChallenge(self) -> None:
    """
        The same L{CramMD5Credentials} will always provide the same challenge,
        no matter how many times it is called.
        """
    c = CramMD5Credentials()
    chal = c.getChallenge()
    self.assertEqual(chal, c.getChallenge())
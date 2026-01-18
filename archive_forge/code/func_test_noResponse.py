import hashlib
from binascii import hexlify
from hmac import HMAC
from twisted.cred.credentials import CramMD5Credentials, IUsernameHashedPassword
from twisted.trial.unittest import TestCase
def test_noResponse(self) -> None:
    """
        When there is no response set, calling C{checkPassword} will return
        L{False}.
        """
    c = CramMD5Credentials()
    self.assertFalse(c.checkPassword(b'secret'))
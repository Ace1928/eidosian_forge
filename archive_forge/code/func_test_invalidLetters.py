from socket import AF_IPX
from twisted.internet.abstract import isIPAddress
from twisted.trial.unittest import TestCase
def test_invalidLetters(self) -> None:
    """
        L{isIPAddress} should return C{False} for any non-decimal dotted
        representation including letters.
        """
    self.assertFalse(isIPAddress('a.2.3.4'))
    self.assertFalse(isIPAddress('1.b.3.4'))
from socket import AF_IPX
from twisted.internet.abstract import isIPAddress
from twisted.trial.unittest import TestCase
def test_invalidNegative(self) -> None:
    """
        L{isIPAddress} should return C{False} for negative decimal values.
        """
    self.assertFalse(isIPAddress('-1'))
    self.assertFalse(isIPAddress('1.-2'))
    self.assertFalse(isIPAddress('1.2.-3'))
    self.assertFalse(isIPAddress('1.2.-3.4'))
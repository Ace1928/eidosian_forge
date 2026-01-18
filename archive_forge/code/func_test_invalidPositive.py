from socket import AF_IPX
from twisted.internet.abstract import isIPAddress
from twisted.trial.unittest import TestCase
def test_invalidPositive(self) -> None:
    """
        L{isIPAddress} should return C{False} for a string containing
        positive decimal values greater than 255.
        """
    self.assertFalse(isIPAddress('256.0.0.0'))
    self.assertFalse(isIPAddress('0.256.0.0'))
    self.assertFalse(isIPAddress('0.0.256.0'))
    self.assertFalse(isIPAddress('0.0.0.256'))
    self.assertFalse(isIPAddress('256.256.256.256'))
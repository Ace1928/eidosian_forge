from socket import AF_IPX
from twisted.internet.abstract import isIPAddress
from twisted.trial.unittest import TestCase
def test_decimalDotted(self) -> None:
    """
        L{isIPAddress} should return C{True} for any decimal dotted
        representation of an IPv4 address.
        """
    self.assertTrue(isIPAddress('0.1.2.3'))
    self.assertTrue(isIPAddress('252.253.254.255'))
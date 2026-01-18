from socket import AF_IPX
from twisted.internet.abstract import isIPAddress
from twisted.trial.unittest import TestCase
def test_nonIPAddressFamily(self) -> None:
    """
        All address families other than C{AF_INET} and C{AF_INET6} result in a
        L{ValueError} being raised.
        """
    self.assertRaises(ValueError, isIPAddress, b'anything', AF_IPX)
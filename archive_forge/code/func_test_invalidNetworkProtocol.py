from twisted.internet import address
from twisted.trial import unittest
from .. import _v1parser
from .._exceptions import InvalidNetworkProtocol, InvalidProxyHeader, MissingAddressData
def test_invalidNetworkProtocol(self) -> None:
    """
        Test that an exception is raised when the proto is not TCP or UNKNOWN.
        """
    self.assertRaises(InvalidNetworkProtocol, _v1parser.V1Parser.parse, b'PROXY WUTPROTO ')
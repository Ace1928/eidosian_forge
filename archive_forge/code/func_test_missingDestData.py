from twisted.internet import address
from twisted.trial import unittest
from .. import _v1parser
from .._exceptions import InvalidNetworkProtocol, InvalidProxyHeader, MissingAddressData
def test_missingDestData(self) -> None:
    """
        Test that an exception is raised when the proto has no destination.
        """
    self.assertRaises(MissingAddressData, _v1parser.V1Parser.parse, b'PROXY TCP4 127.0.0.1 8080 8888')
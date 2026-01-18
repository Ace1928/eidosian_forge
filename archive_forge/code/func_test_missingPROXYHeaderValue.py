from twisted.internet import address
from twisted.trial import unittest
from .. import _v1parser
from .._exceptions import InvalidNetworkProtocol, InvalidProxyHeader, MissingAddressData
def test_missingPROXYHeaderValue(self) -> None:
    """
        Test that an exception is raised when the PROXY header is missing.
        """
    self.assertRaises(InvalidProxyHeader, _v1parser.V1Parser.parse, b'NOTPROXY ')
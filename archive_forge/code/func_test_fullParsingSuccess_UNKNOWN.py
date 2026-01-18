from twisted.internet import address
from twisted.trial import unittest
from .. import _v1parser
from .._exceptions import InvalidNetworkProtocol, InvalidProxyHeader, MissingAddressData
def test_fullParsingSuccess_UNKNOWN(self) -> None:
    """
        Test that parsing is successful for a UNKNOWN PROXY header.
        """
    info = _v1parser.V1Parser.parse(b'PROXY UNKNOWN anything could go here')
    self.assertIsNone(info.source)
    self.assertIsNone(info.destination)
from twisted.internet import address
from twisted.trial import unittest
from .. import _v1parser
from .._exceptions import InvalidNetworkProtocol, InvalidProxyHeader, MissingAddressData
def test_missingSourceData(self) -> None:
    """
        Test that an exception is raised when the proto has no source data.
        """
    self.assertRaises(MissingAddressData, _v1parser.V1Parser.parse, b'PROXY TCP4 ')
from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def test_invalidSignature(self) -> None:
    """
        Test if an invalid signature block raises InvalidProxyError.
        """
    header = _makeHeaderIPv4(sig=b'\x00' * 12)
    self.assertRaises(InvalidProxyHeader, _v2parser.V2Parser.parse, header)
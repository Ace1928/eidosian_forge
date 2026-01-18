from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def test_invalidProto(self) -> None:
    """
        Test if an invalid protocol raises InvalidProxyError.
        """
    header = _makeHeaderIPv4(famProto=b'$')
    self.assertRaises(InvalidProxyHeader, _v2parser.V2Parser.parse, header)
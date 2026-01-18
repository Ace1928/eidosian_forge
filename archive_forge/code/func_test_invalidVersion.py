from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def test_invalidVersion(self) -> None:
    """
        Test if an invalid version raises InvalidProxyError.
        """
    header = _makeHeaderIPv4(verCom=b'\x11')
    self.assertRaises(InvalidProxyHeader, _v2parser.V2Parser.parse, header)
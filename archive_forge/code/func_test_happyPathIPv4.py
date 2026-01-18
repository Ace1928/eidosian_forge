from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def test_happyPathIPv4(self) -> None:
    """
        Test if a well formed IPv4 header is parsed without error.
        """
    header = _makeHeaderIPv4()
    self.assertTrue(_v2parser.V2Parser.parse(header))
from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def test_localCommandIpv6(self) -> None:
    """
        Test that local does not return endpoint data for IPv6 connections.
        """
    header = _makeHeaderIPv6(verCom=b' ')
    info = _v2parser.V2Parser.parse(header)
    self.assertFalse(info.source)
    self.assertFalse(info.destination)
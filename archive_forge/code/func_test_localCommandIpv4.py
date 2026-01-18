from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def test_localCommandIpv4(self) -> None:
    """
        Test that local does not return endpoint data for IPv4 connections.
        """
    header = _makeHeaderIPv4(verCom=b' ')
    info = _v2parser.V2Parser.parse(header)
    self.assertFalse(info.source)
    self.assertFalse(info.destination)
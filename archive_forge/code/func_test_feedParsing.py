from twisted.internet import address
from twisted.trial import unittest
from .. import _v1parser
from .._exceptions import InvalidNetworkProtocol, InvalidProxyHeader, MissingAddressData
def test_feedParsing(self) -> None:
    """
        Test that parsing happens when fed a complete line.
        """
    parser = _v1parser.V1Parser()
    info, remaining = parser.feed(b'PROXY TCP4 127.0.0.1 127.0.0.1 ')
    self.assertFalse(info)
    self.assertFalse(remaining)
    info, remaining = parser.feed(b'8080 8888')
    self.assertFalse(info)
    self.assertFalse(remaining)
    info, remaining = parser.feed(b'\r\n')
    self.assertFalse(remaining)
    assert remaining is not None
    assert info is not None
    self.assertIsInstance(info.source, address.IPv4Address)
    assert isinstance(info.source, address.IPv4Address)
    assert isinstance(info.destination, address.IPv4Address)
    self.assertEqual(info.source.host, '127.0.0.1')
    self.assertEqual(info.source.port, 8080)
    self.assertEqual(info.destination.host, '127.0.0.1')
    self.assertEqual(info.destination.port, 8888)
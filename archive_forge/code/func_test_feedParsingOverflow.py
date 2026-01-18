from twisted.internet import address
from twisted.trial import unittest
from .. import _v1parser
from .._exceptions import InvalidNetworkProtocol, InvalidProxyHeader, MissingAddressData
def test_feedParsingOverflow(self) -> None:
    """
        Test that parsing leaves overflow bytes in the buffer.
        """
    parser = _v1parser.V1Parser()
    info, remaining = parser.feed(b'PROXY TCP4 127.0.0.1 127.0.0.1 8080 8888\r\nHTTP/1.1 GET /\r\n')
    self.assertTrue(info)
    self.assertEqual(remaining, b'HTTP/1.1 GET /\r\n')
    self.assertFalse(parser.buffer)
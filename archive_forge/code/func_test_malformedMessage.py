import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
def test_malformedMessage(self):
    """
        Test that when an unparsable message is received, datagramReceived does
        not raise an exception while processing it.
        """
    unparsable = b'\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x02\x11WWWWWWWWWW-XXXXXX\x08_arduino\x04_tcp\x05local\x00\x00\xff\x80\x01\xc07\x00\x0c\x00\x01\x00\x00\x11\x94\x00\x02\xc0V\xc0V\x00!\x00\x01\x00\x00\x11\x94\x00\x08\x00\x00\x00\x00 J\xc0\x8f\xc0V\x00\x10\x00\x01\x00\x00\x11\x94\x00K\x0eauth_upload=no board="ESP8266_WEMOS_D1MINILITE"\rssh_upload=no\x0ctcp_check=no\xc0\x8f\x00\x01\x00\x01\x00\x00\x00x\x00\x04\xc0\xa8\x01)'
    self.proto.datagramReceived(unparsable, address.IPv4Address('UDP', '127.0.0.1', 12345))
    self.assertEqual(self.controller.messages, [])
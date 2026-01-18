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
def test_emptyQuery(self):
    """
        Test that bytes representing an empty query message can be decoded
        as such.
        """
    msg = dns.Message()
    msg.fromStr(b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
    self.assertEqual(msg.id, 256)
    self.assertFalse(msg.answer, 'Message was not supposed to be an answer.')
    self.assertEqual(msg.opCode, dns.OP_QUERY)
    self.assertFalse(msg.auth, 'Message was not supposed to be authoritative.')
    self.assertFalse(msg.trunc, 'Message was not supposed to be truncated.')
    self.assertEqual(msg.queries, [])
    self.assertEqual(msg.answers, [])
    self.assertEqual(msg.authority, [])
    self.assertEqual(msg.additional, [])
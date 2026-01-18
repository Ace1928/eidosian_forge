from twisted.internet import address, defer, protocol, reactor
from twisted.protocols import portforward, wire
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_QOTD(self):
    """
        Test wire.QOTD protocol.
        """
    t = proto_helpers.StringTransport()
    a = wire.QOTD()
    a.makeConnection(t)
    self.assertEqual(t.value(), b'An apple a day keeps the doctor away.\r\n')
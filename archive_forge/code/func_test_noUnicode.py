from twisted.internet import address, defer, protocol, reactor
from twisted.protocols import portforward, wire
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_noUnicode(self):
    """
        Test that L{proto_helpers.StringTransport} doesn't accept unicode data.
        """
    s = proto_helpers.StringTransport()
    self.assertRaises(TypeError, s.write, 'foo')
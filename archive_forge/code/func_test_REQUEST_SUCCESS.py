import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_REQUEST_SUCCESS(self):
    """
        Test that global request success packets cause the Deferred to be
        called back.
        """
    d = self.conn.sendGlobalRequest(b'request', b'data', True)
    self.conn.ssh_REQUEST_SUCCESS(b'data')

    def check(data):
        self.assertEqual(data, b'data')
    d.addCallback(check)
    d.addErrback(self.fail)
    return d
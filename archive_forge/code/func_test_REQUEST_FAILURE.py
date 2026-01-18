import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_REQUEST_FAILURE(self):
    """
        Test that global request failure packets cause the Deferred to be
        erred back.
        """
    d = self.conn.sendGlobalRequest(b'request', b'data', True)
    self.conn.ssh_REQUEST_FAILURE(b'data')

    def check(f):
        self.assertEqual(f.value.data, b'data')
    d.addCallback(self.fail)
    d.addErrback(check)
    return d
import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_CHANNEL_REQUEST_SUCCESS(self):
    """
        Test that channel request success messages cause the Deferred to be
        called back.
        """
    channel = TestChannel()
    self._openChannel(channel)
    d = self.conn.sendRequest(channel, b'test', b'data', True)
    self.conn.ssh_CHANNEL_SUCCESS(b'\x00\x00\x00\x00')

    def check(result):
        self.assertTrue(result)
    return d
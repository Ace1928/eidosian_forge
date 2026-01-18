import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_CHANNEL_REQUEST_success(self):
    """
        Test that channel requests that succeed send MSG_CHANNEL_SUCCESS.
        """
    channel = TestChannel()
    self._openChannel(channel)
    self.conn.ssh_CHANNEL_REQUEST(b'\x00\x00\x00\x00' + common.NS(b'test') + b'\x00')
    self.assertEqual(channel.numberRequests, 1)
    d = self.conn.ssh_CHANNEL_REQUEST(b'\x00\x00\x00\x00' + common.NS(b'test') + b'\xff' + b'data')

    def check(result):
        self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_SUCCESS, b'\x00\x00\x00\xff')])
    d.addCallback(check)
    return d
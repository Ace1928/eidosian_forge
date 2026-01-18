import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_CHANNEL_CLOSE(self):
    """
        Test that channel close messages are passed up to the channel.  Also,
        test that channel.close() is called if both sides are closed when this
        message is received.
        """
    channel = TestChannel()
    self._openChannel(channel)
    self.assertTrue(channel.gotOpen)
    self.assertFalse(channel.gotOneClose)
    self.assertFalse(channel.gotClosed)
    self.conn.sendClose(channel)
    self.conn.ssh_CHANNEL_CLOSE(b'\x00\x00\x00\x00')
    self.assertTrue(channel.gotOneClose)
    self.assertTrue(channel.gotClosed)
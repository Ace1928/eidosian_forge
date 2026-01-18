import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_CHANNEL_EOF(self):
    """
        Test that channel eof messages are passed up to the channel.
        """
    channel = TestChannel()
    self._openChannel(channel)
    self.conn.ssh_CHANNEL_EOF(b'\x00\x00\x00\x00')
    self.assertTrue(channel.gotEOF)
import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_CHANNEL_OPEN_FAILURE(self):
    """
        Test that channel open failure packets cause the channel to be
        notified that its opening failed.
        """
    channel = TestChannel()
    self.conn.openChannel(channel)
    self.conn.ssh_CHANNEL_OPEN_FAILURE(b'\x00\x00\x00\x00\x00\x00\x00\x01' + common.NS(b'failure!'))
    self.assertEqual(channel.openFailureReason.args, (b'failure!', 1))
    self.assertIsNone(self.conn.channels.get(channel))
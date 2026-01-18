import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_adjustWindow(self):
    """
        Test that channel window adjust messages cause bytes to be added
        to the window.
        """
    channel = TestChannel(localWindow=5)
    self._openChannel(channel)
    channel.localWindowLeft = 0
    self.conn.adjustWindow(channel, 1)
    self.assertEqual(channel.localWindowLeft, 1)
    channel.localClosed = True
    self.conn.adjustWindow(channel, 2)
    self.assertEqual(channel.localWindowLeft, 1)
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_WINDOW_ADJUST, b'\x00\x00\x00\xff\x00\x00\x00\x01')])
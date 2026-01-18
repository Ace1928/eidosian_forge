import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_CHANNEL_EXTENDED_DATA(self):
    """
        Test that channel extended data messages are passed up to the channel,
        or cause the channel to be closed if they're too big.
        """
    channel = TestChannel(localWindow=6, localMaxPacket=5)
    self._openChannel(channel)
    self.conn.ssh_CHANNEL_EXTENDED_DATA(b'\x00\x00\x00\x00\x00\x00\x00\x00' + common.NS(b'data'))
    self.assertEqual(channel.extBuffer, [(0, b'data')])
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_WINDOW_ADJUST, b'\x00\x00\x00\xff\x00\x00\x00\x04')])
    self.transport.packets = []
    longData = b'a' * (channel.localWindowLeft + 1)
    self.conn.ssh_CHANNEL_EXTENDED_DATA(b'\x00\x00\x00\x00\x00\x00\x00\x00' + common.NS(longData))
    self.assertEqual(channel.extBuffer, [(0, b'data')])
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_CLOSE, b'\x00\x00\x00\xff')])
    channel = TestChannel()
    self._openChannel(channel)
    bigData = b'a' * (channel.localMaxPacket + 1)
    self.transport.packets = []
    self.conn.ssh_CHANNEL_EXTENDED_DATA(b'\x00\x00\x00\x01\x00\x00\x00\x00' + common.NS(bigData))
    self.assertEqual(channel.extBuffer, [])
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_CLOSE, b'\x00\x00\x00\xff')])
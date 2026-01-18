import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def channel_ErrorChannel(self, windowSize, maxPacket, data):
    """
        The other side is requesting the ErrorChannel.  Raise an exception.
        """
    raise AssertionError('no such thing')
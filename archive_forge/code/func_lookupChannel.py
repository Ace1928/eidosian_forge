import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def lookupChannel(self, channelType, windowSize, maxPacket, data):
    """
        The server wants us to return a channel.  If the requested channel is
        our TestChannel, return it, otherwise return None.
        """
    if channelType == TestChannel.name:
        return TestChannel(remoteWindow=windowSize, remoteMaxPacket=maxPacket, data=data, avatar=self)
    elif channelType == b'conch-error-args':
        raise error.ConchError(self._ARGS_ERROR_CODE, 'error args in wrong order')
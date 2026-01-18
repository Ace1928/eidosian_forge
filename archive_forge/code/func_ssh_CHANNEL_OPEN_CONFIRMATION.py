import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def ssh_CHANNEL_OPEN_CONFIRMATION(self, packet):
    """
        The other side accepted our MSG_CHANNEL_OPEN request.  Payload::
            uint32  local channel number
            uint32  remote channel number
            uint32  remote window size
            uint32  remote maximum packet size
            <channel specific data>

        Find the channel using the local channel number and notify its
        channelOpen method.
        """
    localChannel, remoteChannel, windowSize, maxPacket = struct.unpack('>4L', packet[:16])
    specificData = packet[16:]
    channel = self.channels[localChannel]
    channel.conn = self
    self.localToRemoteChannel[localChannel] = remoteChannel
    self.channelsToRemoteChannel[channel] = remoteChannel
    channel.remoteWindowLeft = windowSize
    channel.remoteMaxPacket = maxPacket
    channel.channelOpen(specificData)
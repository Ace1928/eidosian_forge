import os
import re
import struct
from unittest import skipIf
from hamcrest import assert_that, equal_to
from twisted.internet import defer
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import StringTransport
from twisted.protocols import loopback
from twisted.python import components
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
def test_clientClosesChannelOnConnnection(self):
    """
        A client sending CHANNEL_CLOSE should trigger closeReceived on the
        associated channel instance.
        """
    conn = self.buildServerConnection()
    packet = common.NS(b'session') + struct.pack('>L', 0) * 3
    conn.ssh_CHANNEL_OPEN(packet)
    sessionChannel = conn.channels[0]
    sessionChannel.request_subsystem(common.NS(b'sftp'))
    sftpServer = sessionChannel.client.transport.proto
    self.interceptConnectionLost(sftpServer)
    self.interceptConnectionLost(sftpServer)
    conn.ssh_CHANNEL_CLOSE(struct.pack('>L', 0))
    self.assertSFTPConnectionLost()
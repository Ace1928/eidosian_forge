import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_heartbeat(self):
    """
        When the I{RPL_WELCOME} message is received a heartbeat is started that
        will send a I{PING} message to the IRC server every
        L{irc.IRCClient.heartbeatInterval} seconds. When the transport is
        closed the heartbeat looping call is stopped too.
        """

    def _createHeartbeat():
        heartbeat = self._originalCreateHeartbeat()
        heartbeat.clock = self.clock
        return heartbeat
    self.clock = task.Clock()
    self._originalCreateHeartbeat = self.client._createHeartbeat
    self.patch(self.client, '_createHeartbeat', _createHeartbeat)
    self.assertIdentical(self.client._heartbeat, None)
    self.client.irc_RPL_WELCOME('foo', [])
    self.assertNotIdentical(self.client._heartbeat, None)
    self.assertEqual(self.client.hostname, 'foo')
    self.assertEqualBufferValue(self.transport.value(), '')
    self.clock.advance(self.client.heartbeatInterval)
    self.assertEqualBufferValue(self.transport.value(), 'PING foo\r\n')
    self.transport.loseConnection()
    self.client.connectionLost(None)
    self.assertEqual(len(self.clock.getDelayedCalls()), 0)
    self.assertIdentical(self.client._heartbeat, None)
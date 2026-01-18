import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
def test_failedExec(self):
    """
        If L{SSHChannel.sendRequest} issues an exec which the server responds to
        with an error, the L{Deferred} it returns fires its errback.
        """
    channel = self._ourServerOurClientTest()

    def cbChannel(channel):
        self.channel = channel
        return self.assertFailure(channel.conn.sendRequest(channel, b'exec', common.NS(b'jumboliah'), 1), Exception)
    channel.addCallback(cbChannel)

    def cbFailed(ignored):
        errors = self.flushLoggedErrors(error.ConchError)
        self.assertEqual(errors[0].value.args, ('bad exec', None))
    channel.addCallback(cbFailed)
    return channel
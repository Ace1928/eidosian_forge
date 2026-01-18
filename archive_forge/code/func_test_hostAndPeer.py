import itertools
import os
import sys
from unittest import skipIf
from twisted.internet import defer, error, protocol, reactor, stdio
from twisted.python import filepath, log
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SkipTest, TestCase
def test_hostAndPeer(self):
    """
        Verify that the transport of a protocol connected to L{StandardIO}
        has C{getHost} and C{getPeer} methods.
        """
    p = StandardIOTestProcessProtocol()
    d = p.onCompletion
    self._spawnProcess(p, b'stdio_test_hostpeer')

    def processEnded(reason):
        host, peer = p.data[1].splitlines()
        self.assertTrue(host)
        self.assertTrue(peer)
        reason.trap(error.ProcessDone)
    return self._requireFailure(d, processEnded)
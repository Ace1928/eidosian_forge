import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def test_ControlBackslash(self):
    """
        Evaluate cancelling with CTRL-\\.
        """
    self._testwrite(b'cancelled line')
    partialLine = self.recvlineClient.expect(b'cancelled line')

    def gotPartialLine(ign):
        self._assertBuffer([b'>>> cancelled line'])
        self._testwrite(manhole.CTRL_BACKSLASH)
        d = self.recvlineClient.onDisconnection
        return self.assertFailure(d, error.ConnectionDone)

    def gotClearedLine(ign):
        self._assertBuffer([b''])
    return partialLine.addCallback(gotPartialLine).addCallback(gotClearedLine)
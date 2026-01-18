import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def test_interruptDuringContinuation(self):
    """
        Sending ^C to Manhole while in a state where more input is required to
        complete a statement should discard the entire ongoing statement and
        reset the input prompt to the non-continuation prompt.
        """
    continuing = self.recvlineClient.expect(b'things')
    self._testwrite(b'(\nthings')

    def gotContinuation(ignored):
        self._assertBuffer([b'>>> (', b'... things'])
        interrupted = self.recvlineClient.expect(b'>>> ')
        self._testwrite(manhole.CTRL_C)
        return interrupted
    continuing.addCallback(gotContinuation)

    def gotInterruption(ignored):
        self._assertBuffer([b'>>> (', b'... things', b'KeyboardInterrupt', b'>>> '])
    continuing.addCallback(gotInterruption)
    return continuing
import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def test_ControlC(self):
    """
        Evaluate interrupting with CTRL-C.
        """
    done = self.recvlineClient.expect(b'done')
    self._testwrite(b'cancelled line' + manhole.CTRL_C + b'done')

    def finished(ign):
        self._assertBuffer([b'>>> cancelled line', b'KeyboardInterrupt', b'>>> done'])
    return done.addCallback(finished)
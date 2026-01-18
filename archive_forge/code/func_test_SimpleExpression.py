import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def test_SimpleExpression(self):
    """
        Evaluate simple expression.
        """
    done = self.recvlineClient.expect(b'done')
    self._testwrite(b'1 + 1\ndone')

    def finished(ign):
        self._assertBuffer([b'>>> 1 + 1', b'2', b'>>> done'])
    return done.addCallback(finished)
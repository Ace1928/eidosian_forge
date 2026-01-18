import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def test_FunctionDefinition(self):
    """
        Evaluate function definition.
        """
    done = self.recvlineClient.expect(b'done')
    self._testwrite(b'def foo(bar):\n\tprint(bar)\n\nfoo(42)\ndone')

    def finished(ign):
        self._assertBuffer([b'>>> def foo(bar):', b'...     print(bar)', b'... ', b'>>> foo(42)', b'42', b'>>> done'])
    return done.addCallback(finished)
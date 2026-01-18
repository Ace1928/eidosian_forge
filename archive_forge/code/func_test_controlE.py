import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def test_controlE(self):
    """
        CTRL-E can be used as END - setting cursor to end of current
        line buffer.
        """
    self._testwrite(b'rint "hello' + b'\x01' + b'p' + b'\x05' + b'"')
    d = self.recvlineClient.expect(b'print "hello"')

    def cb(ignore):
        self._assertBuffer([b'>>> print "hello"'])
    return d.addCallback(cb)
import os
import sys
from unittest import skipIf
from twisted.conch import recvline
from twisted.conch.insults import insults
from twisted.cred import portal
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.python import components, filepath, reflect
from twisted.python.compat import iterbytes
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
from twisted.conch import telnet
from twisted.conch.insults import helper
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import checkers
from twisted.conch.test import test_telnet
def test_home(self):
    """
        When L{HistoricRecvLine} receives a HOME keystroke it moves the
        cursor to the beginning of the current line buffer.
        """
    kR = lambda ch: self.p.keystrokeReceived(ch, None)
    for ch in iterbytes(b'hello, world'):
        kR(ch)
    self.assertEqual(self.p.currentLineBuffer(), (b'hello, world', b''))
    kR(self.pt.HOME)
    self.assertEqual(self.p.currentLineBuffer(), (b'', b'hello, world'))
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
def test_horizontalArrows(self):
    """
        When L{HistoricRecvLine} receives a LEFT_ARROW or
        RIGHT_ARROW keystroke it moves the cursor left or right
        in the current line buffer, respectively.
        """
    kR = lambda ch: self.p.keystrokeReceived(ch, None)
    for ch in iterbytes(b'xyz'):
        kR(ch)
    self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
    kR(self.pt.RIGHT_ARROW)
    self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
    kR(self.pt.LEFT_ARROW)
    self.assertEqual(self.p.currentLineBuffer(), (b'xy', b'z'))
    kR(self.pt.LEFT_ARROW)
    self.assertEqual(self.p.currentLineBuffer(), (b'x', b'yz'))
    kR(self.pt.LEFT_ARROW)
    self.assertEqual(self.p.currentLineBuffer(), (b'', b'xyz'))
    kR(self.pt.LEFT_ARROW)
    self.assertEqual(self.p.currentLineBuffer(), (b'', b'xyz'))
    kR(self.pt.RIGHT_ARROW)
    self.assertEqual(self.p.currentLineBuffer(), (b'x', b'yz'))
    kR(self.pt.RIGHT_ARROW)
    self.assertEqual(self.p.currentLineBuffer(), (b'xy', b'z'))
    kR(self.pt.RIGHT_ARROW)
    self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
    kR(self.pt.RIGHT_ARROW)
    self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
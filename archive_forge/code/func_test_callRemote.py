import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def test_callRemote(self):
    """
        L{CommandDispatcher.callRemote} should emit a properly formatted '_ask'
        box to its boxSender and record an outstanding L{Deferred}.  When a
        corresponding '_answer' packet is received, the L{Deferred} should be
        fired, and the results translated via the given L{Command}'s response
        de-serialization.
        """
    D = self.dispatcher.callRemote(Hello, hello=b'world')
    self.assertEqual(self.sender.sentBoxes, [amp.AmpBox(_command=b'hello', _ask=b'1', hello=b'world')])
    answers = []
    D.addCallback(answers.append)
    self.assertEqual(answers, [])
    self.dispatcher.ampBoxReceived(amp.AmpBox({b'hello': b'yay', b'print': b'ignored', b'_answer': b'1'}))
    self.assertEqual(answers, [dict(hello=b'yay', Print='ignored')])
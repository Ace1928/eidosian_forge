from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def test_clientGreetingSequence(self):
    """
        Like C{test_clientGreeting}, but use C{writeSequence} instead of
        C{write} to issue the greeting.
        """
    return self._greetingtest('writeSequence', False)
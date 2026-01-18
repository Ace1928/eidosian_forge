from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def cbComplete(ignored):
    self.assertEqual(pumpCalls, [(client, [b'baz', b'quux', None]), (server, [b'foo', b'bar'])])
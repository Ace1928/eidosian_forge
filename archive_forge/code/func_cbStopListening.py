import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
def cbStopListening(ignored):
    del serverFactory.stopFactory
    port.startListening()
    client = MyClientFactory()
    serverFactory.protocolConnectionMade = defer.Deferred()
    client.protocolConnectionMade = defer.Deferred()
    connector = reactor.connectTCP('127.0.0.1', port.getHost().port, client)
    self.addCleanup(connector.disconnect)
    return defer.gatherResults([serverFactory.protocolConnectionMade, client.protocolConnectionMade]).addCallback(close)
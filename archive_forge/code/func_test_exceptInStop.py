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
def test_exceptInStop(self):
    """
        If the server factory raises an exception in C{stopFactory}, the
        deferred returned by L{tcp.Port.stopListening} should fail with the
        corresponding error.
        """
    serverFactory = MyServerFactory()

    def raiseException():
        raise RuntimeError('An error')
    serverFactory.stopFactory = raiseException
    port = reactor.listenTCP(0, serverFactory, interface='127.0.0.1')
    return self.assertFailure(port.stopListening(), RuntimeError)
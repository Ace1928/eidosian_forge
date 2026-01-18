import errno
import gc
import io
import os
import socket
from functools import wraps
from typing import Callable, ClassVar, List, Mapping, Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass, verifyObject
import attr
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import (
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import (
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.test.test_tcp import (
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
def assertPeerClosedOnEMFILE(testCase, exhauster, reactor, runReactor, listen, connect):
    """
    Assert that an L{IListeningPort} immediately closes an accepted
    peer socket when the number of open file descriptors exceeds the
    soft resource limit.

    @param testCase: The test case under which to run this assertion.
    @type testCase: L{trial.unittest.SynchronousTestCase}

    @param exhauster: The file descriptor exhauster.
    @type exhauster: L{_ExhaustsFileDescriptors}

    @param reactor: The reactor under test.

    @param runReactor: A callable that will synchronously run the
        provided reactor.

    @param listen: A callback to bind to a port.
    @type listen: A L{callable} that accepts two arguments: the
        provided C{reactor}; and a L{ServerFactory}.  It must return
        an L{IListeningPort} provider.

    @param connect: A callback to connect a client to the listening
        port.
    @type connect: A L{callable} that accepts three arguments: the
        provided C{reactor}; the address returned by
        L{IListeningPort.getHost}; and a L{ClientFactory}.  Its return
        value is ignored.
    """
    testCase.addCleanup(exhauster.release)
    serverFactory = MyServerFactory()
    serverConnectionMade = Deferred()
    serverFactory.protocolConnectionMade = serverConnectionMade
    serverConnectionCompleted = [False]

    def stopReactorIfServerAccepted(_):
        reactor.stop()
        serverConnectionCompleted[0] = True
    serverConnectionMade.addCallback(stopReactorIfServerAccepted)
    clientFactory = MyClientFactory()
    if IReactorTime.providedBy(reactor):

        def inner():
            port = listen(reactor, serverFactory)
            listeningHost = port.getHost()
            connect(reactor, listeningHost, clientFactory)
            exhauster.exhaust()
        reactor.callLater(0, reactor.callLater, 0, inner)
    else:
        port = listen(reactor, serverFactory)
        listeningHost = port.getHost()
        connect(reactor, listeningHost, clientFactory)
        reactor.callWhenRunning(exhauster.exhaust)

    def stopReactorAndCloseFileDescriptors(result):
        exhauster.release()
        reactor.stop()
        return result
    clientFactory.deferred.addBoth(stopReactorAndCloseFileDescriptors)
    clientFactory.failDeferred.addBoth(stopReactorAndCloseFileDescriptors)
    runReactor(reactor)
    noResult = []
    serverConnectionMade.addBoth(noResult.append)
    testCase.assertFalse(noResult, 'Server accepted connection; EMFILE not triggered.')
    testCase.assertNoResult(clientFactory.failDeferred)
    testCase.successResultOf(clientFactory.deferred)
    testCase.assertRaises(ConnectionClosed, clientFactory.lostReason.raiseException)
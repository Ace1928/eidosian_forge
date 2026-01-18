import os
import socket
import traceback
from unittest import skipIf
from zope.interface import implementer
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet, IReadDescriptor
from twisted.internet.tcp import EINPROGRESS, EWOULDBLOCK
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
def test_connectionLostOnShutdown(self):
    """
        Any file descriptors added to the reactor have their C{connectionLost}
        called when C{reactor.stop} is called.
        """
    reactor = self.buildReactor()

    class DoNothingDescriptor(FileDescriptor):

        def doRead(self):
            return None

        def doWrite(self):
            return None
    client, server = self._connectedPair()
    fd1 = DoNothingDescriptor(reactor)
    fd1.fileno = client.fileno
    fd2 = DoNothingDescriptor(reactor)
    fd2.fileno = server.fileno
    reactor.addReader(fd1)
    reactor.addWriter(fd2)
    reactor.callWhenRunning(reactor.stop)
    self.runReactor(reactor)
    self.assertTrue(fd1.disconnected)
    self.assertTrue(fd2.disconnected)
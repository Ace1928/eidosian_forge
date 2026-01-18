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
def test_negativeOneFileDescriptor(self):
    """
        If L{FileDescriptor.fileno} returns C{-1}, the descriptor is removed
        from the reactor.
        """
    reactor = self.buildReactor()
    client, server = self._connectedPair()

    class DisappearingDescriptor(FileDescriptor):
        _fileno = server.fileno()
        _received = b''

        def fileno(self):
            return self._fileno

        def doRead(self):
            self._fileno = -1
            self._received += server.recv(1)
            client.send(b'y')

        def connectionLost(self, reason):
            reactor.stop()
    descriptor = DisappearingDescriptor(reactor)
    reactor.addReader(descriptor)
    client.send(b'x')
    self.runReactor(reactor)
    self.assertEqual(descriptor._received, b'x')
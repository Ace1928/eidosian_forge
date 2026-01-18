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
@skipIf(platform.isWindows(), 'Cannot duplicate socket filenos on Windows')
def test_lostFileDescriptor(self):
    """
        The file descriptor underlying a FileDescriptor may be closed and
        replaced by another at some point.  Bytes which arrive on the new
        descriptor must not be delivered to the FileDescriptor which was
        originally registered with the original descriptor of the same number.

        Practically speaking, this is difficult or impossible to detect.  The
        implementation relies on C{fileno} raising an exception if the original
        descriptor has gone away.  If C{fileno} continues to return the original
        file descriptor value, the reactor may deliver events from that
        descriptor.  This is a best effort attempt to ease certain debugging
        situations.  Applications should not rely on it intentionally.
        """
    reactor = self.buildReactor()
    name = reactor.__class__.__name__
    if name in ('EPollReactor', 'KQueueReactor', 'CFReactor', 'AsyncioSelectorReactor'):
        raise SkipTest(f'{name!r} cannot detect lost file descriptors')
    client, server = self._connectedPair()

    class Victim(FileDescriptor):
        """
            This L{FileDescriptor} will have its socket closed out from under it
            and another socket will take its place.  It will raise a
            socket.error from C{fileno} after this happens (because socket
            objects remember whether they have been closed), so as long as the
            reactor calls the C{fileno} method the problem will be detected.
            """

        def fileno(self):
            return server.fileno()

        def doRead(self):
            raise Exception('Victim.doRead should never be called')

        def connectionLost(self, reason):
            """
                When the problem is detected, the reactor should disconnect this
                file descriptor.  When that happens, stop the reactor so the
                test ends.
                """
            reactor.stop()
    reactor.addReader(Victim())

    def messItUp():
        newC, newS = self._connectedPair()
        fileno = server.fileno()
        server.close()
        os.dup2(newS.fileno(), fileno)
        newC.send(b'x')
    reactor.callLater(0, messItUp)
    self.runReactor(reactor)
    self.flushLoggedErrors(socket.error)
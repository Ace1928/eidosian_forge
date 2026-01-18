import errno
import socket
from zope.interface import verify
from twisted.internet.error import UnsupportedAddressFamily
from twisted.internet.interfaces import IReactorSocket
from twisted.internet.protocol import DatagramProtocol, ServerFactory
from twisted.internet.test.reactormixins import ReactorBuilder, needsRunningReactor
from twisted.python.log import err
from twisted.python.runtime import platform
def test_invalidDescriptor(self):
    """
        An implementation of L{IReactorSocket.adoptDatagramPort} raises
        L{socket.error} if passed an integer which is not associated with a
        socket.
        """
    reactor = self.buildReactor()
    probe = socket.socket()
    fileno = probe.fileno()
    probe.close()
    exc = self.assertRaises(socket.error, reactor.adoptDatagramPort, fileno, socket.AF_INET, DatagramProtocol())
    if platform.isWindows():
        self.assertEqual(exc.args[0], errno.WSAENOTSOCK)
    else:
        self.assertEqual(exc.args[0], errno.EBADF)
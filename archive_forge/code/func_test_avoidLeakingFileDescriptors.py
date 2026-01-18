from hashlib import md5
from os import close, fstat, stat, unlink, urandom
from pprint import pformat
from socket import AF_INET, SOCK_STREAM, SOL_SOCKET, socket
from stat import S_IMODE
from struct import pack
from tempfile import mkstemp, mktemp
from typing import Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from twisted.internet import base, interfaces
from twisted.internet.address import UNIXAddress
from twisted.internet.defer import Deferred, fail, gatherResults
from twisted.internet.endpoints import UNIXClientEndpoint, UNIXServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, DatagramProtocol, ServerFactory
from twisted.internet.task import LoopingCall
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.log import addObserver, err, removeObserver
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
@skipIf(not sendmsg, sendmsgSkipReason)
def test_avoidLeakingFileDescriptors(self):
    """
        If associated with a protocol which does not provide
        L{IFileDescriptorReceiver}, file descriptors received by the
        L{IUNIXTransport} implementation are closed and a warning is emitted.
        """
    from socket import socketpair
    probeClient, probeServer = socketpair()
    events = []
    addObserver(events.append)
    self.addCleanup(removeObserver, events.append)

    class RecordEndpointAddresses(SendFileDescriptor):

        def connectionMade(self):
            self.hostAddress = self.transport.getHost()
            self.peerAddress = self.transport.getPeer()
            SendFileDescriptor.connectionMade(self)
    server = RecordEndpointAddresses(probeClient.fileno(), b'junk')
    client = ConnectableProtocol()
    runProtocolsWithReactor(self, server, client, self.endpoints)
    probeClient.close()
    probeServer.setblocking(False)
    self.assertEqual(b'', probeServer.recv(1024))
    format = '%(protocolName)s (on %(hostAddress)r) does not provide IFileDescriptorReceiver; closing file descriptor received (from %(peerAddress)r).'
    clsName = 'ConnectableProtocol'
    expectedEvent = dict(hostAddress=server.peerAddress, peerAddress=server.hostAddress, protocolName=clsName, format=format)
    for logEvent in events:
        for k, v in expectedEvent.items():
            if v != logEvent.get(k):
                break
        else:
            break
    else:
        self.fail('Expected event (%s) not found in logged events (%s)' % (expectedEvent, pformat(events)))
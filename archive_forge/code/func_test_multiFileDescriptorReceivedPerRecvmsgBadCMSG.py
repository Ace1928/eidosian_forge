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
def test_multiFileDescriptorReceivedPerRecvmsgBadCMSG(self):
    """
        _SendmsgMixin handles multiple file descriptors per recvmsg, calling
        L{IFileDescriptorReceiver.fileDescriptorReceived} once per received
        file descriptor. Scenario: unsupported CMSGs.
        """
    from twisted.python import sendmsg

    def ancillaryPacker(fdsToSend):
        ancillary = []
        expectedCount = 0
        return (ancillary, expectedCount)

    def fakeRecvmsgUnsupportedAncillary(skt, *args, **kwargs):
        data = b'some data'
        ancillary = [(None, None, b'')]
        flags = 0
        return sendmsg.ReceivedMessage(data, ancillary, flags)
    events = []
    addObserver(events.append)
    self.addCleanup(removeObserver, events.append)
    self.patch(sendmsg, 'recvmsg', fakeRecvmsgUnsupportedAncillary)
    self._sendmsgMixinFileDescriptorReceivedDriver(ancillaryPacker)
    expectedMessage = 'received unsupported ancillary data'
    found = any((expectedMessage in e['format'] for e in events))
    self.assertTrue(found, 'Expected message not found in logged events')
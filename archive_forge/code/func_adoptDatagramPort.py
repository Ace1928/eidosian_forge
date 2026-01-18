from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
def adoptDatagramPort(fileDescriptor: int, addressFamily: 'AddressFamily', protocol: 'DatagramProtocol', maxPacketSize: int) -> 'IListeningPort':
    """
        Add an existing listening I{SOCK_DGRAM} socket to the reactor to
        monitor for read and write readiness.

        @param fileDescriptor: A file descriptor associated with a socket which
            is already bound to an address and marked as listening.  The socket
            must be set non-blocking.  Any additional flags (for example,
            close-on-exec) must also be set by application code.  Application
            code is responsible for closing the file descriptor, which may be
            done as soon as C{adoptDatagramPort} returns.
        @param addressFamily: The address family or I{domain} of the socket.
            For example, L{socket.AF_INET6}.
        @param protocol: A L{DatagramProtocol} instance to connect to
            a UDP transport.
        @param maxPacketSize: The maximum packet size to accept.

        @return: An object providing L{IListeningPort}.

        @raise UnsupportedAddressFamily: If the given address family is not
            supported by this reactor, or not supported with the given socket
            type.
        @raise UnsupportedSocketType: If the given socket type is not supported
            by this reactor, or not supported with the given socket type.
        """
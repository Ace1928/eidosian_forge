import os
import socket
import sys
from struct import unpack
from typing import Tuple
from twisted.python.sendmsg import recvmsg
def recvfd(socketfd: int) -> Tuple[int, bytes]:
    """
    Receive a file descriptor from a L{sendmsg} message on the given C{AF_UNIX}
    socket.

    @param socketfd: An C{AF_UNIX} socket, attached to another process waiting
        to send sockets via the ancillary data mechanism in L{send1msg}.

    @param fd: C{int}

    @return: a 2-tuple of (new file descriptor, description).
    @rtype: 2-tuple of (C{int}, C{bytes})
    """
    ourSocket = socket.fromfd(socketfd, socket.AF_UNIX, socket.SOCK_STREAM)
    data, ancillary, flags = recvmsg(ourSocket)
    [(cmsgLevel, cmsgType, packedFD)] = ancillary
    [unpackedFD] = unpack('i', packedFD)
    return (unpackedFD, data)
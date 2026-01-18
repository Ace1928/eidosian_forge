fromfd() -- create a socket object from an open file descriptor [*]
fromshare() -- create a socket object from data received from socket.share() [*]
import _socket
from _socket import *
import os, sys, io, selectors
from enum import IntEnum, IntFlag
def recv_fds(sock, bufsize, maxfds, flags=0):
    """ recv_fds(sock, bufsize, maxfds[, flags]) -> (data, list of file
        descriptors, msg_flags, address)

        Receive up to maxfds file descriptors returning the message
        data and a list containing the descriptors.
        """
    fds = array.array('i')
    msg, ancdata, flags, addr = sock.recvmsg(bufsize, _socket.CMSG_LEN(maxfds * fds.itemsize))
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == _socket.SOL_SOCKET and cmsg_type == _socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[:len(cmsg_data) - len(cmsg_data) % fds.itemsize])
    return (msg, list(fds), flags, addr)
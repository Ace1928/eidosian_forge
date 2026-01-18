import array
import os
import socket
from warnings import warn
@classmethod
def from_ancdata(cls, ancdata) -> ['FileDescriptor']:
    """Make a list of FileDescriptor from received file descriptors

        ancdata is a list of ancillary data tuples as returned by socket.recvmsg()
        """
    fds = array.array('i')
    for cmsg_level, cmsg_type, data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(data[:len(data) - len(data) % fds.itemsize])
    return [cls(i) for i in fds]
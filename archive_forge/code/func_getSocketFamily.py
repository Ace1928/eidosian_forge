from collections import namedtuple
from socket import CMSG_SPACE, SCM_RIGHTS, socket as Socket
from typing import List, Tuple
def getSocketFamily(socket: Socket) -> int:
    """
    Return the family of the given socket.

    @param socket: The socket to get the family of.
    """
    return socket.family
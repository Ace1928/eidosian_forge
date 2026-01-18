import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def makeport(self):
    """Create a new socket and send a PORT command for it."""
    sock = socket.create_server(('', 0), family=self.af, backlog=1)
    port = sock.getsockname()[1]
    host = self.sock.getsockname()[0]
    if self.af == socket.AF_INET:
        resp = self.sendport(host, port)
    else:
        resp = self.sendeprt(host, port)
    if self.timeout is not _GLOBAL_DEFAULT_TIMEOUT:
        sock.settimeout(self.timeout)
    return sock
import os
import sys
import time
import warnings
import contextlib
import portend
@property
def bound_addr(self):
    """
        The bind address, or if it's an ephemeral port and the
        socket has been bound, return the actual port bound.
        """
    host, port = self.bind_addr
    if port == 0 and self.httpserver.socket:
        port = self.httpserver.socket.getsockname()[1]
    return (host, port)
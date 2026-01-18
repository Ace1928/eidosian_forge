import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def getwelcome(self):
    """Get the welcome message from the server.
        (this is read and squirreled away by connect())"""
    if self.debugging:
        print('*welcome*', self.sanitize(self.welcome))
    return self.welcome
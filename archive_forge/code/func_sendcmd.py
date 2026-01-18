import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def sendcmd(self, cmd):
    """Send a command and return the response."""
    self.putcmd(cmd)
    return self.getresp()
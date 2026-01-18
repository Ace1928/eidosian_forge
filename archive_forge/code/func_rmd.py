import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def rmd(self, dirname):
    """Remove a directory."""
    return self.voidcmd('RMD ' + dirname)
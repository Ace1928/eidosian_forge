import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def prot_c(self):
    """Set up clear text data connection."""
    resp = self.voidcmd('PROT C')
    self._prot_p = False
    return resp
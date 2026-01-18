import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def ccc(self):
    """Switch back to a clear-text control connection."""
    if not isinstance(self.sock, ssl.SSLSocket):
        raise ValueError('not using TLS')
    resp = self.voidcmd('CCC')
    self.sock = self.sock.unwrap()
    return resp
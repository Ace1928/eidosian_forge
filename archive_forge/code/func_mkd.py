import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def mkd(self, dirname):
    """Make a directory, return its full pathname."""
    resp = self.voidcmd('MKD ' + dirname)
    if not resp.startswith('257'):
        return ''
    return parse257(resp)
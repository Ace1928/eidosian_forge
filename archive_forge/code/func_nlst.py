import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def nlst(self, *args):
    """Return a list of files in a given directory (default the current)."""
    cmd = 'NLST'
    for arg in args:
        cmd = cmd + (' ' + arg)
    files = []
    self.retrlines(cmd, files.append)
    return files
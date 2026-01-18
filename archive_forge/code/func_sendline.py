import socket
from contextlib import contextmanager
from .exceptions import TIMEOUT, EOF
from .spawnbase import SpawnBase
def sendline(self, s) -> int:
    """Write to socket with trailing newline, return number of bytes written"""
    s = self._coerce_send_string(s)
    return self.send(s + self.linesep)
import copy
import functools
import socket
import struct
import time
from typing import Any, Optional
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import dns.inet
def have(self, amount):
    if len(self._buffer) >= amount:
        return True
    if self._seen_end:
        raise UnexpectedEOF
    return False
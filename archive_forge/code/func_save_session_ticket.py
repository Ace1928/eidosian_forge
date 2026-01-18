import selectors
import socket
import ssl
import struct
import threading
import time
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import aioquic.quic.events  # type: ignore
import dns.exception
import dns.inet
from dns.quic._common import (
def save_session_ticket(self, address, port, ticket):
    with self._lock:
        super().save_session_ticket(address, port, ticket)
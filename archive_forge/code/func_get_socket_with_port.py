import os
import socket
from string import Template
from typing import List, Any
def get_socket_with_port() -> socket.socket:
    addrs = socket.getaddrinfo(host='localhost', port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
    for addr in addrs:
        family, type, proto, _, _ = addr
        s = socket.socket(family, type, proto)
        try:
            s.bind(('localhost', 0))
            s.listen(0)
            return s
        except OSError as e:
            s.close()
    raise RuntimeError('Failed to create a socket')
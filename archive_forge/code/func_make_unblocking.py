from __future__ import (absolute_import, division, print_function)
import fcntl
import os
import os.path
import socket as pysocket
from ansible.module_utils.six import PY2
def make_unblocking(sock):
    if hasattr(sock, '_sock'):
        sock._sock.setblocking(0)
    elif hasattr(sock, 'setblocking'):
        sock.setblocking(0)
    else:
        make_file_unblocking(sock)
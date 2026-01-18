from __future__ import (absolute_import, division, print_function)
import fcntl
import os
import os.path
import socket as pysocket
from ansible.module_utils.six import PY2
def shutdown_writing(sock, log=_empty_writer):
    if hasattr(sock, 'shutdown_write'):
        sock.shutdown_write()
    elif hasattr(sock, 'shutdown'):
        try:
            sock.shutdown(pysocket.SHUT_WR)
        except TypeError as e:
            log('Shutting down for writing not possible; trying shutdown instead: {0}'.format(e))
            sock.shutdown()
    elif not PY2 and isinstance(sock, getattr(pysocket, 'SocketIO')):
        sock._sock.shutdown(pysocket.SHUT_WR)
    else:
        log('No idea how to signal end of writing')
import errno
import os
import socket
import sys
from pdb import Pdb
from billiard.process import current_process
def get_avail_port(self, host, port, search_limit=100, skew=+0):
    try:
        _, skew = current_process().name.split('-')
        skew = int(skew)
    except ValueError:
        pass
    this_port = None
    for i in range(search_limit):
        _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        this_port = port + skew + i
        try:
            _sock.bind((host, this_port))
        except OSError as exc:
            if exc.errno in [errno.EADDRINUSE, errno.EINVAL]:
                continue
            raise
        else:
            return (_sock, this_port)
    raise Exception(NO_AVAILABLE_PORT.format(self=self))
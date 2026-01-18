import errno
import os
import select
import socket as pysocket
import struct
def frames_iter_tty(socket):
    """
    Return a generator of data read from the socket when the tty setting is
    enabled.
    """
    while True:
        result = read(socket)
        if len(result) == 0:
            return
        yield result
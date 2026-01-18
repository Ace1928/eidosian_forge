import errno
import os
import select
import socket as pysocket
import struct
def frames_iter_no_tty(socket):
    """
    Returns a generator of data read from the socket when the tty setting is
    not enabled.
    """
    while True:
        stream, n = next_frame_header(socket)
        if n < 0:
            break
        while n > 0:
            result = read(socket, n)
            if result is None:
                continue
            data_length = len(result)
            if data_length == 0:
                return
            n -= data_length
            yield (stream, result)
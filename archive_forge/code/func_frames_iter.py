import errno
import os
import select
import socket as pysocket
import struct
def frames_iter(socket, tty):
    """
    Return a generator of frames read from socket. A frame is a tuple where
    the first item is the stream number and the second item is a chunk of data.

    If the tty setting is enabled, the streams are multiplexed into the stdout
    stream.
    """
    if tty:
        return ((STDOUT, frame) for frame in frames_iter_tty(socket))
    else:
        return frames_iter_no_tty(socket)
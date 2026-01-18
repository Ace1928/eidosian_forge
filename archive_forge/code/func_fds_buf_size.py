import array
import os
import socket
from warnings import warn
def fds_buf_size():
    global _fds_buf_size_cache
    if _fds_buf_size_cache is None:
        maxfds = 256
        fd_size = array.array('i').itemsize
        _fds_buf_size_cache = socket.CMSG_SPACE(maxfds * fd_size)
    return _fds_buf_size_cache
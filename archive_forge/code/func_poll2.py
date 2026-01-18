import select
import socket
import sys
import time
import warnings
import os
from errno import EALREADY, EINPROGRESS, EWOULDBLOCK, ECONNRESET, EINVAL, \
def poll2(timeout=0.0, map=None):
    if map is None:
        map = socket_map
    if timeout is not None:
        timeout = int(timeout * 1000)
    pollster = select.poll()
    if map:
        for fd, obj in list(map.items()):
            flags = 0
            if obj.readable():
                flags |= select.POLLIN | select.POLLPRI
            if obj.writable() and (not obj.accepting):
                flags |= select.POLLOUT
            if flags:
                pollster.register(fd, flags)
        r = pollster.poll(timeout)
        for fd, flags in r:
            obj = map.get(fd)
            if obj is None:
                continue
            readwrite(obj, flags)
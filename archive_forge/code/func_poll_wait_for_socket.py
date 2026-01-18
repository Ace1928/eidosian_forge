import errno
import select
import sys
from functools import partial
def poll_wait_for_socket(sock, read=False, write=False, timeout=None):
    if not read and (not write):
        raise RuntimeError('must specify at least one of read=True, write=True')
    mask = 0
    if read:
        mask |= select.POLLIN
    if write:
        mask |= select.POLLOUT
    poll_obj = select.poll()
    poll_obj.register(sock, mask)

    def do_poll(t):
        if t is not None:
            t *= 1000
        return poll_obj.poll(t)
    return bool(_retry_on_intr(do_poll, timeout))
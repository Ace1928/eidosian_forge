import os
import sys
import stat
import select
import time
import errno
def poll_ignore_interrupts(fds, timeout=None):
    """Simple wrapper around poll to register file descriptors and
    ignore signals."""
    if timeout is not None:
        end_time = time.time() + timeout
    poller = select.poll()
    for fd in fds:
        poller.register(fd, select.POLLIN | select.POLLPRI | select.POLLHUP | select.POLLERR)
    while True:
        try:
            timeout_ms = None if timeout is None else timeout * 1000
            results = poller.poll(timeout_ms)
            return [afd for afd, _ in results]
        except InterruptedError:
            err = sys.exc_info()[1]
            if err.args[0] == errno.EINTR:
                if timeout is not None:
                    timeout = end_time - time.time()
                    if timeout < 0:
                        return []
            else:
                raise
import os
import errno
from pty import (STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO, CHILD)
from .util import PtyProcessError
def pty_make_controlling_tty(tty_fd):
    """This makes the pseudo-terminal the controlling tty. This should be
    more portable than the pty.fork() function. Specifically, this should
    work on Solaris. """
    child_name = os.ttyname(tty_fd)
    try:
        fd = os.open('/dev/tty', os.O_RDWR | os.O_NOCTTY)
        os.close(fd)
    except OSError as err:
        if err.errno != errno.ENXIO:
            raise
    os.setsid()
    try:
        fd = os.open('/dev/tty', os.O_RDWR | os.O_NOCTTY)
        os.close(fd)
        raise PtyProcessError('OSError of errno.ENXIO should be raised.')
    except OSError as err:
        if err.errno != errno.ENXIO:
            raise
    fd = os.open(child_name, os.O_RDWR)
    os.close(fd)
    fd = os.open('/dev/tty', os.O_WRONLY)
    os.close(fd)
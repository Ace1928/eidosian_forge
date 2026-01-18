import os
import errno
from pty import (STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO, CHILD)
from .util import PtyProcessError
def fork_pty():
    """This implements a substitute for the forkpty system call. This
    should be more portable than the pty.fork() function. Specifically,
    this should work on Solaris.

    Modified 10.06.05 by Geoff Marshall: Implemented __fork_pty() method to
    resolve the issue with Python's pty.fork() not supporting Solaris,
    particularly ssh. Based on patch to posixmodule.c authored by Noah
    Spurrier::

        http://mail.python.org/pipermail/python-dev/2003-May/035281.html

    """
    parent_fd, child_fd = os.openpty()
    if parent_fd < 0 or child_fd < 0:
        raise OSError('os.openpty() failed')
    pid = os.fork()
    if pid == CHILD:
        os.close(parent_fd)
        pty_make_controlling_tty(child_fd)
        os.dup2(child_fd, STDIN_FILENO)
        os.dup2(child_fd, STDOUT_FILENO)
        os.dup2(child_fd, STDERR_FILENO)
    else:
        os.close(child_fd)
    return (pid, parent_fd)
from select import select
import os
import sys
import tty
from os import close, waitpid
from tty import setraw, tcgetattr, tcsetattr
def slave_open(tty_name):
    """slave_open(tty_name) -> slave_fd
    Open the pty slave and acquire the controlling terminal, returning
    opened filedescriptor.
    Deprecated, use openpty() instead."""
    result = os.open(tty_name, os.O_RDWR)
    try:
        from fcntl import ioctl, I_PUSH
    except ImportError:
        return result
    try:
        ioctl(result, I_PUSH, 'ptem')
        ioctl(result, I_PUSH, 'ldterm')
    except OSError:
        pass
    return result
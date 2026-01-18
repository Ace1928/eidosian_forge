import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
def close_fds(*fds):
    """Close each file descriptor given as an argument"""
    for fd in fds:
        os.close(fd)
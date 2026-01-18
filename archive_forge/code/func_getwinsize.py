import os
import sys
import time
import pty
import tty
import errno
import signal
from contextlib import contextmanager
import ptyprocess
from ptyprocess.ptyprocess import use_native_pty_fork
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .spawnbase import SpawnBase
from .utils import (
def getwinsize(self):
    """This returns the terminal window size of the child tty. The return
        value is a tuple of (rows, cols). """
    return self.ptyproc.getwinsize()
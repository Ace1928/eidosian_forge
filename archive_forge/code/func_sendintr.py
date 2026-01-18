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
def sendintr(self):
    """This sends a SIGINT to the child. It does not require
        the SIGINT to be the first character on a line. """
    n, byte = self.ptyproc.sendintr()
    self._log_control(byte)
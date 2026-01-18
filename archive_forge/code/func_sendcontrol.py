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
def sendcontrol(self, char):
    """Helper method that wraps send() with mnemonic access for sending control
        character to the child (such as Ctrl-C or Ctrl-D).  For example, to send
        Ctrl-G (ASCII 7, bell, '\x07')::

            child.sendcontrol('g')

        See also, sendintr() and sendeof().
        """
    n, byte = self.ptyproc.sendcontrol(char)
    self._log_control(byte)
    return n
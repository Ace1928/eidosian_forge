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
@flag_eof.setter
def flag_eof(self, value):
    self.ptyproc.flag_eof = value
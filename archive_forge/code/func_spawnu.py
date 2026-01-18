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
def spawnu(*args, **kwargs):
    """Deprecated: pass encoding to spawn() instead."""
    kwargs.setdefault('encoding', 'utf-8')
    return spawn(*args, **kwargs)
from __future__ import annotations
import bz2
import errno
import gzip
import io
import mmap
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING
def get_open_fds() -> int:
    """
    Return the number of open file descriptors for current process

    Warnings:
        Will only work on UNIX-like OS-es.
    """
    pid: int = os.getpid()
    procs: bytes = subprocess.check_output(['lsof', '-w', '-Ff', '-p', str(pid)])
    _procs: str = procs.decode('utf-8')
    return len([s for s in _procs.split('\n') if s and s[0] == 'f' and s[1:].isdigit()])
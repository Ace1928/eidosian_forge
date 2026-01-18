from __future__ import annotations
import atexit
from contextlib import contextmanager
import fnmatch
import importlib.util
import io
import os
from os.path import abspath
from os.path import dirname
from os.path import exists
from os.path import isabs
from os.path import isdir
from os.path import isfile
from os.path import islink
from os.path import normpath
import posixpath
from stat import S_ISDIR
from stat import S_ISLNK
from stat import S_ISREG
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Literal
from typing import overload
from typing import TYPE_CHECKING
import uuid
import warnings
from . import error
def sysexec(self, *argv: os.PathLike[str], **popen_opts: Any) -> str:
    """Return stdout text from executing a system child process,
        where the 'self' path points to executable.
        The process is directly invoked and not through a system shell.
        """
    from subprocess import PIPE
    from subprocess import Popen
    popen_opts.pop('stdout', None)
    popen_opts.pop('stderr', None)
    proc = Popen([str(self)] + [str(arg) for arg in argv], **popen_opts, stdout=PIPE, stderr=PIPE)
    stdout: str | bytes
    stdout, stderr = proc.communicate()
    ret = proc.wait()
    if isinstance(stdout, bytes):
        stdout = stdout.decode(sys.getdefaultencoding())
    if ret != 0:
        if isinstance(stderr, bytes):
            stderr = stderr.decode(sys.getdefaultencoding())
        raise RuntimeError(ret, ret, str(self), stdout, stderr)
    return stdout
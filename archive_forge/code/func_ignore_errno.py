import atexit
import errno
import math
import numbers
import os
import platform as _platform
import signal as _signal
import sys
import warnings
from contextlib import contextmanager
from billiard.compat import close_open_fds, get_fdmax
from billiard.util import set_pdeathsig as _set_pdeathsig
from kombu.utils.compat import maybe_fileno
from kombu.utils.encoding import safe_str
from .exceptions import SecurityError, SecurityWarning, reraise
from .local import try_import
@contextmanager
def ignore_errno(*errnos, **kwargs):
    """Context manager to ignore specific POSIX error codes.

    Takes a list of error codes to ignore: this can be either
    the name of the code, or the code integer itself::

        >>> with ignore_errno('ENOENT'):
        ...     with open('foo', 'r') as fh:
        ...         return fh.read()

        >>> with ignore_errno(errno.ENOENT, errno.EPERM):
        ...    pass

    Arguments:
        types (Tuple[Exception]): A tuple of exceptions to ignore
            (when the errno matches).  Defaults to :exc:`Exception`.
    """
    types = kwargs.get('types') or (Exception,)
    errnos = [get_errno_name(errno) for errno in errnos]
    try:
        yield
    except types as exc:
        if not hasattr(exc, 'errno'):
            raise
        if exc.errno not in errnos:
            raise
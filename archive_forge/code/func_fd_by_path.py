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
def fd_by_path(paths):
    """Return a list of file descriptors.

    This method returns list of file descriptors corresponding to
    file paths passed in paths variable.

    Arguments:
        paths: List[str]: List of file paths.

    Returns:
        List[int]: List of file descriptors.

    Example:
        >>> keep = fd_by_path(['/dev/urandom', '/my/precious/'])
    """
    stats = set()
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            stats.add(os.fstat(fd)[1:3])
        finally:
            os.close(fd)

    def fd_in_stats(fd):
        try:
            return os.fstat(fd)[1:3] in stats
        except OSError:
            return False
    return [_fd for _fd in range(get_fdmax(2048)) if fd_in_stats(_fd)]
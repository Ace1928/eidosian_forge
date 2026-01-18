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
def setgroups(groups):
    """Set active groups from a list of group ids."""
    max_groups = None
    try:
        max_groups = os.sysconf('SC_NGROUPS_MAX')
    except Exception:
        pass
    try:
        return _setgroups_hack(groups[:max_groups])
    except OSError as exc:
        if exc.errno != errno.EPERM:
            raise
        if any((group not in groups for group in os.getgroups())):
            raise
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
def parse_uid(uid):
    """Parse user id.

    Arguments:
        uid (str, int): Actual uid, or the username of a user.
    Returns:
        int: The actual uid.
    """
    try:
        return int(uid)
    except ValueError:
        try:
            return pwd.getpwnam(uid).pw_uid
        except (AttributeError, KeyError):
            raise KeyError(f'User does not exist: {uid}')
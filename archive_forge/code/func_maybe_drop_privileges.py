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
def maybe_drop_privileges(uid=None, gid=None):
    """Change process privileges to new user/group.

    If UID and GID is specified, the real user/group is changed.

    If only UID is specified, the real user is changed, and the group is
    changed to the users primary group.

    If only GID is specified, only the group is changed.
    """
    if sys.platform == 'win32':
        return
    if os.geteuid():
        if not os.getuid():
            raise SecurityError('contact support')
    uid = uid and parse_uid(uid)
    gid = gid and parse_gid(gid)
    if uid:
        _setuid(uid, gid)
    else:
        gid and setgid(gid)
    if uid and (not os.getuid()) and (not os.geteuid()):
        raise SecurityError('Still root uid after drop privileges!')
    if gid and (not os.getgid()) and (not os.getegid()):
        raise SecurityError('Still root gid after drop privileges!')
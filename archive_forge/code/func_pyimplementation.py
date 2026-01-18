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
def pyimplementation():
    """Return string identifying the current Python implementation."""
    if hasattr(_platform, 'python_implementation'):
        return _platform.python_implementation()
    elif sys.platform.startswith('java'):
        return 'Jython ' + sys.platform
    elif hasattr(sys, 'pypy_version_info'):
        v = '.'.join((str(p) for p in sys.pypy_version_info[:3]))
        if sys.pypy_version_info[3:]:
            v += '-' + ''.join((str(p) for p in sys.pypy_version_info[3:]))
        return 'PyPy ' + v
    else:
        return 'CPython'
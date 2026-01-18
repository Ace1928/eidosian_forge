import codecs
import numbers
import os
import platform
import re
import subprocess
import sys
from humanfriendly.compat import coerce_string, is_unicode, on_windows, which
from humanfriendly.decorators import cached
from humanfriendly.deprecation import define_aliases
from humanfriendly.text import concatenate, format
from humanfriendly.usage import format_usage
@cached
def enable_ansi_support():
    """
    Try to enable support for ANSI escape sequences (required on Windows).

    :returns: :data:`True` if ANSI is supported, :data:`False` otherwise.

    This functions checks for the following supported configurations, in the
    given order:

    1. On Windows, if :func:`have_windows_native_ansi_support()` confirms
       native support for ANSI escape sequences :mod:`ctypes` will be used to
       enable this support.

    2. On Windows, if the environment variable ``$ANSICON`` is set nothing is
       done because it is assumed that support for ANSI escape sequences has
       already been enabled via `ansicon <https://github.com/adoxa/ansicon>`_.

    3. On Windows, an attempt is made to import and initialize the Python
       package :pypi:`colorama` instead (of course for this to work
       :pypi:`colorama` has to be installed).

    4. On other platforms this function calls :func:`connected_to_terminal()`
       to determine whether ANSI escape sequences are supported (that is to
       say all platforms that are not Windows are assumed to support ANSI
       escape sequences natively, without weird contortions like above).

       This makes it possible to call :func:`enable_ansi_support()`
       unconditionally without checking the current platform.

    The :func:`~humanfriendly.decorators.cached` decorator is used to ensure
    that this function is only executed once, but its return value remains
    available on later calls.
    """
    if have_windows_native_ansi_support():
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
        ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-12), 7)
        return True
    elif on_windows():
        if 'ANSICON' in os.environ:
            return True
        try:
            import colorama
            colorama.init()
            return True
        except ImportError:
            return False
    else:
        return connected_to_terminal()
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
def find_terminal_size():
    """
    Determine the number of lines and columns visible in the terminal.

    :returns: A tuple of two integers with the line and column count.

    The result of this function is based on the first of the following three
    methods that works:

    1. First :func:`find_terminal_size_using_ioctl()` is tried,
    2. then :func:`find_terminal_size_using_stty()` is tried,
    3. finally :data:`DEFAULT_LINES` and :data:`DEFAULT_COLUMNS` are returned.

    .. note:: The :func:`find_terminal_size()` function performs the steps
              above every time it is called, the result is not cached. This is
              because the size of a virtual terminal can change at any time and
              the result of :func:`find_terminal_size()` should be correct.

              `Pre-emptive snarky comment`_: It's possible to cache the result
              of this function and use :mod:`signal.SIGWINCH <signal>` to
              refresh the cached values!

              Response: As a library I don't consider it the role of the
              :mod:`humanfriendly.terminal` module to install a process wide
              signal handler ...

    .. _Pre-emptive snarky comment: http://blogs.msdn.com/b/oldnewthing/archive/2008/01/30/7315957.aspx
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            result = find_terminal_size_using_ioctl(stream)
            if min(result) >= 1:
                return result
        except Exception:
            pass
    try:
        result = find_terminal_size_using_stty()
        if min(result) >= 1:
            return result
    except Exception:
        pass
    return (DEFAULT_LINES, DEFAULT_COLUMNS)
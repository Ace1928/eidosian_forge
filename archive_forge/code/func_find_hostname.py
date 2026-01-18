import collections
import logging
import os
import re
import socket
import sys
from humanfriendly import coerce_boolean
from humanfriendly.compat import coerce_string, is_string, on_windows
from humanfriendly.terminal import ANSI_COLOR_CODES, ansi_wrap, enable_ansi_support, terminal_supports_colors
from humanfriendly.text import format, split
def find_hostname(use_chroot=True):
    """
    Find the host name to include in log messages.

    :param use_chroot: Use the name of the chroot when inside a chroot?
                       (boolean, defaults to :data:`True`)
    :returns: A suitable host name (a string).

    Looks for :data:`CHROOT_FILES` that have a nonempty first line (taken to be
    the chroot name). If none are found then :func:`socket.gethostname()` is
    used as a fall back.
    """
    for chroot_file in CHROOT_FILES:
        try:
            with open(chroot_file) as handle:
                first_line = next(handle)
                name = first_line.strip()
                if name:
                    return name
        except Exception:
            pass
    return socket.gethostname()
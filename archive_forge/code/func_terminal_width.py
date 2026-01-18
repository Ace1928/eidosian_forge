import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def terminal_width():
    """Return terminal width.

    None is returned if the width can't established precisely.

    The rules are:
    - if BRZ_COLUMNS is set, returns its value
    - if there is no controlling terminal, returns None
    - query the OS, if the queried size has changed since the last query,
      return its value,
    - if COLUMNS is set, returns its value,
    - if the OS has a value (even though it's never changed), return its value.

    From there, we need to query the OS to get the size of the controlling
    terminal.

    On Unices we query the OS by:
    - get termios.TIOCGWINSZ
    - if an error occurs or a negative value is obtained, returns None

    On Windows we query the OS by:
    - win32utils.get_console_size() decides,
    - returns None on error (provided default value)
    """
    try:
        width = int(os.environ['BRZ_COLUMNS'])
    except (KeyError, ValueError):
        width = None
    if width is not None:
        if width > 0:
            return width
        else:
            return None
    isatty = getattr(sys.stdout, 'isatty', None)
    if isatty is None or not isatty():
        return None
    width, height = os_size = _terminal_size(None, None)
    global _first_terminal_size, _terminal_size_state
    if _terminal_size_state == 'no_data':
        _first_terminal_size = os_size
        _terminal_size_state = 'unchanged'
    elif _terminal_size_state == 'unchanged' and _first_terminal_size != os_size:
        _terminal_size_state = 'changed'
    if _terminal_size_state == 'changed':
        if width is not None and width > 0:
            return width
    try:
        return int(os.environ['COLUMNS'])
    except (KeyError, ValueError):
        pass
    if _terminal_size_state == 'unchanged':
        if width is not None and width > 0:
            return width
    return None
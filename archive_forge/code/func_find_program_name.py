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
def find_program_name():
    """
    Select a suitable program name to embed in log messages.

    :returns: One of the following strings (in decreasing order of preference):

              1. The base name of the currently running Python program or
                 script (based on the value at index zero of :data:`sys.argv`).
              2. The base name of the Python executable (based on
                 :data:`sys.executable`).
              3. The string 'python'.
    """
    return (os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] != '-c' else '') or (os.path.basename(sys.executable) if sys.executable else '') or 'python'
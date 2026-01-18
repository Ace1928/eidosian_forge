from contextlib import contextmanager
import curses
from curses import setupterm, tigetnum, tigetstr, tparm
from fcntl import ioctl
from six import text_type, string_types
from os import isatty, environ
import struct
import sys
from termios import TIOCGWINSZ
@property
def on_color(self):
    """Return a capability that sets the background color.

        See ``color()``.

        """
    return ParametrizingString(self._background_color, self.normal)
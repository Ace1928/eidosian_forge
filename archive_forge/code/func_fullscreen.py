from contextlib import contextmanager
import curses
from curses import setupterm, tigetnum, tigetstr, tparm
from fcntl import ioctl
from six import text_type, string_types
from os import isatty, environ
import struct
import sys
from termios import TIOCGWINSZ
@contextmanager
def fullscreen(self):
    """Return a context manager that enters fullscreen mode while inside it
        and restores normal mode on leaving."""
    self.stream.write(self.enter_fullscreen)
    try:
        yield
    finally:
        self.stream.write(self.exit_fullscreen)
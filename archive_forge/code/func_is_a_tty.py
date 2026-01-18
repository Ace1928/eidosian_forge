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
def is_a_tty(self):
    """Whether my ``stream`` appears to be associated with a terminal"""
    return self._is_a_tty
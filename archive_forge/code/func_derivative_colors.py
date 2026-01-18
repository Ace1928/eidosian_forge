from contextlib import contextmanager
import curses
from curses import setupterm, tigetnum, tigetstr, tparm
from fcntl import ioctl
from six import text_type, string_types
from os import isatty, environ
import struct
import sys
from termios import TIOCGWINSZ
def derivative_colors(colors):
    """Return the names of valid color variants, given the base colors."""
    return set(['on_' + c for c in colors] + ['bright_' + c for c in colors] + ['on_bright_' + c for c in colors])
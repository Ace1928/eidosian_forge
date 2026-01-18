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
def does_styling(self):
    """Whether attempt to emit capabilities

        This is influenced by the ``is_a_tty`` property and by the
        ``force_styling`` argument to the constructor. You can examine
        this value to decide whether to draw progress bars or other frippery.

        """
    return self._does_styling
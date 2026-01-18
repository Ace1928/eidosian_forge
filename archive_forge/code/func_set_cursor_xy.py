from fcntl import ioctl
import re
import six
import struct
import sys
from termios import TIOCGWINSZ, TCSADRAIN, tcsetattr, tcgetattr
import textwrap
import tty
from .prefs import Prefs
def set_cursor_xy(self, xpos, ypos):
    """
        Set the cursor x, y coordinates.
        @param xpos: The x coordinate of the cursor.
        @type xpos: int
        @param ypos: The y coordinate of the cursor.
        @type ypos: int
        """
    self.escape('%d;%dH' % (ypos, xpos))
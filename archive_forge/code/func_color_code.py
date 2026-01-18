import os
from fnmatch import fnmatch
from datetime import datetime
import operator
import re
def color_code(foreground=None, background=None):
    """
    0  black
    1  red
    2  green
    3  yellow
    4  blue
    5  magenta (purple)
    6  cyan
    7  white (gray)

    Add 8 to get high-intensity
    """
    if foreground is None and background is None:
        return '\x1b[0m'
    codes = []
    if foreground is None:
        codes.append('[39m')
    elif foreground > 7:
        codes.append('[1m')
        codes.append('[%im' % (22 + foreground))
    else:
        codes.append('[%im' % (30 + foreground))
    if background is None:
        codes.append('[49m')
    else:
        codes.append('[%im' % (40 + background))
    return '\x1b' + '\x1b'.join(codes)
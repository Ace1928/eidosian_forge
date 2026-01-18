import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@classmethod
def bk_green(cls):
    """Make the text background color green."""
    wAttributes = cls._get_text_attributes()
    wAttributes &= ~win32.BACKGROUND_MASK
    wAttributes |= win32.BACKGROUND_GREEN
    cls._set_text_attributes(wAttributes)
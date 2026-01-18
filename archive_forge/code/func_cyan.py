import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@classmethod
def cyan(cls):
    """Make the text foreground color cyan."""
    wAttributes = cls._get_text_attributes()
    wAttributes &= ~win32.FOREGROUND_MASK
    wAttributes |= win32.FOREGROUND_CYAN
    cls._set_text_attributes(wAttributes)
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@classmethod
def bk_black(cls):
    """Make the text background color black."""
    wAttributes = cls._get_text_attributes()
    wAttributes &= ~win32.BACKGROUND_MASK
    cls._set_text_attributes(wAttributes)
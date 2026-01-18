import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@classmethod
def can_use_colors(cls):
    """
        Determine if we can use colors.

        Colored output only works when the output is a real console, and fails
        when redirected to a file or pipe. Call this method before issuing a
        call to any other method of this class to make sure it's actually
        possible to use colors.

        @rtype:  bool
        @return: C{True} if it's possible to output text with color,
            C{False} otherwise.
        """
    try:
        cls._get_text_attributes()
        return True
    except Exception:
        return False
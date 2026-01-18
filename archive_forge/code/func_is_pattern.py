import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@staticmethod
def is_pattern(token):
    """
        Determine if the given argument is a valid hexadecimal pattern to be
        used with L{pattern}.

        @type  token: str
        @param token: String to parse.

        @rtype:  bool
        @return:
            C{True} if it's a valid hexadecimal pattern, C{False} otherwise.
        """
    return re.match('^(?:[\\?A-Fa-f0-9][\\?A-Fa-f0-9]\\s*)+$', token)
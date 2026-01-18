import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
def log_exc(self):
    """
        Log lines of text associated with the last Python exception.
        """
    self.__do_log('Exception raised: %s' % traceback.format_exc())
import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@staticmethod
def pathname_to_filename(pathname):
    """
        Equivalent to: C{PathOperations.split_filename(pathname)[0]}

        @note: This function is preserved for backwards compatibility with
            WinAppDbg 1.4 and earlier. It may be removed in future versions.

        @type  pathname: str
        @param pathname: Absolute path to a file.

        @rtype:  str
        @return: Filename component of the path.
        """
    return win32.PathFindFileName(pathname)
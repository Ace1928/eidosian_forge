import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@staticmethod
def split_filename(pathname):
    """
        @type  pathname: str
        @param pathname: Absolute path.

        @rtype:  tuple( str, str )
        @return: Tuple containing the path to the file and the base filename.
        """
    filepart = win32.PathFindFileName(pathname)
    pathpart = win32.PathRemoveFileSpec(pathname)
    return (pathpart, filepart)
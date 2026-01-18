import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@staticmethod
def split_extension(pathname):
    """
        @type  pathname: str
        @param pathname: Absolute path.

        @rtype:  tuple( str, str )
        @return:
            Tuple containing the file and extension components of the filename.
        """
    filepart = win32.PathRemoveExtension(pathname)
    extpart = win32.PathFindExtension(pathname)
    return (filepart, extpart)
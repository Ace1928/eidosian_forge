import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@classmethod
def string_list_file(cls, filename, values):
    """
        Write a list of strings to a file.
        If a file of the same name exists, it's contents are replaced.

        See L{HexInput.string_list_file} for a description of the file format.

        @type  filename: str
        @param filename: Name of the file to write.

        @type  values: list( int )
        @param values: List of strings to write to the file.
        """
    fd = open(filename, 'w')
    for string in values:
        (print >> fd, string)
    fd.close()
from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump, HexInput
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses
from winappdbg.module import Module, _ModuleContainer
from winappdbg.thread import Thread, _ThreadContainer
from winappdbg.window import Window
from winappdbg.search import Search, \
from winappdbg.disasm import Disassembler
import re
import os
import os.path
import ctypes
import struct
import warnings
import traceback
@staticmethod
def argv_to_cmdline(argv):
    """
        Convert a list of arguments to a single command line string.

        @type  argv: list( str )
        @param argv: List of argument strings.
            The first element is the program to execute.

        @rtype:  str
        @return: Command line string.
        """
    cmdline = list()
    for token in argv:
        if not token:
            token = '""'
        else:
            if '"' in token:
                token = token.replace('"', '\\"')
            if ' ' in token or '\t' in token or '\n' in token or ('\r' in token):
                token = '"%s"' % token
        cmdline.append(token)
    return ' '.join(cmdline)
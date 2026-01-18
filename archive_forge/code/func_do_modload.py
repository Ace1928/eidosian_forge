from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.util import PathOperations
from winappdbg.event import EventHandler, NoEvent
from winappdbg.textio import HexInput, HexOutput, HexDump, CrashDump, DebugLog
import os
import sys
import code
import time
import warnings
import traceback
from cmd import Cmd
def do_modload(self, arg):
    """
        [~process] modload <filename.dll> - load a DLL module
        """
    filename = self.split_tokens(arg, 1, 1)[0]
    process = self.get_process_from_prefix()
    try:
        process.inject_dll(filename, bWait=False)
    except RuntimeError:
        print("Can't inject module: %r" % filename)
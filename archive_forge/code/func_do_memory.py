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
def do_memory(self, arg):
    """
        [~process] m - show the process memory map
        [~process] memory - show the process memory map
        """
    if arg:
        raise CmdError('too many arguments')
    process = self.get_process_from_prefix()
    try:
        memoryMap = process.get_memory_map()
        mappedFilenames = process.get_mapped_filenames()
        print('')
        print(CrashDump.dump_memory_map(memoryMap, mappedFilenames))
    except WindowsError:
        msg = "can't get memory information for process (%d)"
        raise CmdError(msg % process.get_pid())
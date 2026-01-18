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
def input_command_line(self, command_line):
    argv = self.debug.system.cmdline_to_argv(command_line)
    if not argv:
        raise CmdError('missing command line to execute')
    fname = argv[0]
    if not os.path.exists(fname):
        try:
            fname, _ = win32.SearchPath(None, fname, '.exe')
        except WindowsError:
            raise CmdError('file not found: %s' % fname)
        argv[0] = fname
        command_line = self.debug.system.argv_to_cmdline(argv)
    return command_line
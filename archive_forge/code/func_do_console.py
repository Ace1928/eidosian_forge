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
def do_console(self, arg):
    """
        console <target> [arguments...] - run a console program for debugging
        """
    if self.cmdprefix:
        raise CmdError('prefix not allowed')
    cmdline = self.input_command_line(arg)
    try:
        process = self.debug.execl(arg, bConsole=True, bFollow=self.options.follow)
        print('Spawned process (%d)' % process.get_pid())
    except Exception:
        raise CmdError("can't execute")
    self.set_fake_last_event(process)
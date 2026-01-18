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
def do_processlist(self, arg):
    """
        pl - show the processes being debugged
        processlist - show the processes being debugged
        """
    if self.cmdprefix:
        raise CmdError('prefix not allowed')
    if arg:
        raise CmdError('too many arguments')
    system = self.debug.system
    pid_list = self.debug.get_debugee_pids()
    if pid_list:
        print('Process ID   File name')
        for pid in pid_list:
            if pid == 0:
                filename = 'System Idle Process'
            elif pid == 4:
                filename = 'System'
            else:
                filename = system.get_process(pid).get_filename()
                filename = PathOperations.pathname_to_filename(filename)
            print('%-12d %s' % (pid, filename))
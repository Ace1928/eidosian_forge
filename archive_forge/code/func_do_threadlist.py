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
def do_threadlist(self, arg):
    """
        tl - show the threads being debugged
        threadlist - show the threads being debugged
        """
    if arg:
        raise CmdError('too many arguments')
    if self.cmdprefix:
        process = self.get_process_from_prefix()
        for thread in process.iter_threads():
            tid = thread.get_tid()
            name = thread.get_name()
            print('%-12d %s' % (tid, name))
    else:
        system = self.debug.system
        pid_list = self.debug.get_debugee_pids()
        if pid_list:
            print('Thread ID    Thread name')
            for pid in pid_list:
                process = system.get_process(pid)
                for thread in process.iter_threads():
                    tid = thread.get_tid()
                    name = thread.get_name()
                    print('%-12d %s' % (tid, name))
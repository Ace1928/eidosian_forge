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
def kill_thread(self, tid):
    thread = self.debug.system.get_thread(tid)
    try:
        thread.kill()
        process = thread.get_process()
        pid = process.get_pid()
        if self.debug.is_debugee(pid) and (not process.is_alive()):
            self.debug.detach(pid)
        print('Killed thread (%d)' % tid)
    except Exception:
        print('Error trying to kill thread (%d)' % tid)
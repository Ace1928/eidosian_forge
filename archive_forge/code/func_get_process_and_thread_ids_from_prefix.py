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
def get_process_and_thread_ids_from_prefix(self):
    if self.cmdprefix:
        try:
            pid = self.input_process(self.cmdprefix)
            tid = None
        except CmdError:
            try:
                tid = self.input_thread(self.cmdprefix)
                pid = self.debug.system.get_thread(tid).get_pid()
            except CmdError:
                msg = 'unknown process or thread (%s)' % self.cmdprefix
                raise CmdError(msg)
    else:
        if self.lastEvent is None:
            raise CmdError('no current process set')
        pid = self.lastEvent.get_pid()
        tid = self.lastEvent.get_tid()
    return (pid, tid)
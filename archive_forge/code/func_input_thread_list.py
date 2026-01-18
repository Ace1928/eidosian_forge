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
def input_thread_list(self, token_list):
    targets = set()
    system = self.debug.system
    for token in token_list:
        try:
            tid = self.input_integer(token)
            if not system.has_thread(tid):
                raise CmdError('thread not found (%d)' % tid)
            targets.add(tid)
        except ValueError:
            found = set()
            for process in system.iter_processes():
                found.update(system.find_threads_by_name(token))
            if not found:
                raise CmdError('thread not found (%s)' % token)
            for thread in found:
                targets.add(thread.get_tid())
    targets = list(targets)
    targets.sort()
    return targets
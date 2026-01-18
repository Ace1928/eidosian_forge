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
def input_process_list(self, token_list):
    targets = set()
    system = self.debug.system
    for token in token_list:
        try:
            pid = self.input_integer(token)
            if not system.has_process(pid):
                raise CmdError('process not found (%d)' % pid)
            targets.add(pid)
        except ValueError:
            found = system.find_processes_by_filename(token)
            if not found:
                raise CmdError('process not found (%s)' % token)
            for process, _ in found:
                targets.add(process.get_pid())
    targets = list(targets)
    targets.sort()
    return targets
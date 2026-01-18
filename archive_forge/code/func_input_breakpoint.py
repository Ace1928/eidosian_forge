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
def input_breakpoint(self, token_list):
    pid, tid, address, size = self.input_full_address_range(token_list)
    if not self.debug.is_debugee(pid):
        raise CmdError('target process is not being debugged')
    return (pid, tid, address, size)
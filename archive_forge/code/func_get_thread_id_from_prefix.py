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
def get_thread_id_from_prefix(self):
    if self.cmdprefix:
        tid = self.input_thread(self.cmdprefix)
    else:
        if self.lastEvent is None:
            raise CmdError('no current process set')
        tid = self.lastEvent.get_tid()
    return tid
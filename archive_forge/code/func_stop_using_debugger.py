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
def stop_using_debugger(self):
    if hasattr(self, 'debug'):
        debug = self.debug
        debug.set_event_handler(self.prevHandler)
        del self.prevHandler
        del self.debug
        return debug
    return None
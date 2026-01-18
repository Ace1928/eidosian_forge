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
def do_dd(self, arg):
    """
        [~thread] dd <register> - show memory contents as dwords
        [~thread] dd <register-register> - show memory contents as dwords
        [~thread] dd <register> <size> - show memory contents as dwords
        [~process] dd <address> - show memory contents as dwords
        [~process] dd <address-address> - show memory contents as dwords
        [~process] dd <address> <size> - show memory contents as dwords
        """
    self.print_memory_display(arg, HexDump.hexblock_dword)
    self.last_display_command = self.do_dd
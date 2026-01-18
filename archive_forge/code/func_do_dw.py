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
def do_dw(self, arg):
    """
        [~thread] dw <register> - show memory contents as words
        [~thread] dw <register-register> - show memory contents as words
        [~thread] dw <register> <size> - show memory contents as words
        [~process] dw <address> - show memory contents as words
        [~process] dw <address-address> - show memory contents as words
        [~process] dw <address> <size> - show memory contents as words
        """
    self.print_memory_display(arg, HexDump.hexblock_word)
    self.last_display_command = self.do_dw
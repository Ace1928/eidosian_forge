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
def do_d(self, arg):
    """
        [~thread] d <register> - show memory contents
        [~thread] d <register-register> - show memory contents
        [~thread] d <register> <size> - show memory contents
        [~process] d <address> - show memory contents
        [~process] d <address-address> - show memory contents
        [~process] d <address> <size> - show memory contents
        """
    return self.last_display_command(arg)
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
def do_du(self, arg):
    """
        [~thread] du <register> - show memory contents as Unicode string
        [~process] du <address> - show memory contents as Unicode string
        """
    if not arg:
        arg = self.default_display_target
    token_list = self.split_tokens(arg, 1, 2)
    pid, tid, address, size = self.input_display(token_list, 256)
    process = self.get_process(pid)
    data = process.peek_string(address, True, size)
    if data:
        print(repr(data))
    self.last_display_command = self.do_du
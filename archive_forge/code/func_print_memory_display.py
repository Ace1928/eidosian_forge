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
def print_memory_display(self, arg, method):
    if not arg:
        arg = self.default_display_target
    token_list = self.split_tokens(arg, 1, 2)
    pid, tid, address, size = self.input_display(token_list)
    label = self.get_process(pid).get_label_at_address(address)
    data = self.read_memory(address, size, pid)
    if data:
        print('%s:' % label)
        print(method(data, address))
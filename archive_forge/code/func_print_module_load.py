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
def print_module_load(self, event):
    mod = event.get_module()
    base = mod.get_base()
    name = mod.get_filename()
    if not name:
        name = ''
    msg = 'Loaded module (%s) %s'
    msg = msg % (HexDump.address(base), name)
    print(msg)
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
def is_register(self, token):
    if win32.arch == 'i386':
        if token in self.register_aliases_full_32:
            return True
        token = token.title()
        for name, typ in win32.CONTEXT._fields_:
            if name == token:
                return win32.sizeof(typ) == win32.sizeof(win32.DWORD)
    elif win32.arch == 'amd64':
        if token in self.register_aliases_full_64:
            return True
        token = token.title()
        for name, typ in win32.CONTEXT._fields_:
            if name == token:
                return win32.sizeof(typ) == win32.sizeof(win32.DWORD64)
    return False
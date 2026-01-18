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
def input_register(self, token, tid=None):
    if tid is None:
        if self.lastEvent is None:
            raise CmdError('no current process set')
        thread = self.lastEvent.get_thread()
    else:
        thread = self.debug.system.get_thread(tid)
    ctx = thread.get_context()
    token = token.lower()
    title = token.title()
    if title in ctx:
        return ctx.get(title)
    if ctx.arch == 'i386':
        if token in self.segment_names:
            return ctx.get('Seg%s' % title)
        if token in self.register_alias_32_to_16:
            return ctx.get(self.register_alias_32_to_16[token]) & 65535
        if token in self.register_alias_32_to_8_low:
            return ctx.get(self.register_alias_32_to_8_low[token]) & 255
        if token in self.register_alias_32_to_8_high:
            return (ctx.get(self.register_alias_32_to_8_high[token]) & 65280) >> 8
    elif ctx.arch == 'amd64':
        if token in self.segment_names:
            return ctx.get('Seg%s' % title)
        if token in self.register_alias_64_to_32:
            return ctx.get(self.register_alias_64_to_32[token]) & 4294967295
        if token in self.register_alias_64_to_16:
            return ctx.get(self.register_alias_64_to_16[token]) & 65535
        if token in self.register_alias_64_to_8_low:
            return ctx.get(self.register_alias_64_to_8_low[token]) & 255
        if token in self.register_alias_64_to_8_high:
            return (ctx.get(self.register_alias_64_to_8_high[token]) & 65280) >> 8
    return None
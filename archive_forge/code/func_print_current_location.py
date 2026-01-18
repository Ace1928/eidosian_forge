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
def print_current_location(self, process=None, thread=None, pc=None):
    if not process:
        if self.lastEvent is None:
            raise CmdError('no current process set')
        process = self.lastEvent.get_process()
    if not thread:
        if self.lastEvent is None:
            raise CmdError('no current process set')
        thread = self.lastEvent.get_thread()
    thread.suspend()
    try:
        if pc is None:
            pc = thread.get_pc()
        ctx = thread.get_context()
    finally:
        thread.resume()
    label = process.get_label_at_address(pc)
    try:
        disasm = process.disassemble(pc, 15)
    except WindowsError:
        disasm = None
    except NotImplementedError:
        disasm = None
    print('')
    print(CrashDump.dump_registers(ctx))
    print('%s:' % label)
    if disasm:
        print(CrashDump.dump_code_line(disasm[0], pc, bShowDump=True))
    else:
        try:
            data = process.peek(pc, 15)
        except Exception:
            data = None
        if data:
            print('%s: %s' % (HexDump.address(pc), HexDump.hexblock_byte(data)))
        else:
            print('%s: ???' % HexDump.address(pc))
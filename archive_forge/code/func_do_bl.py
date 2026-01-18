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
def do_bl(self, arg):
    """
        bl - list the breakpoints for the current process
        bl * - list the breakpoints for all processes
        [~process] bl - list the breakpoints for the given process
        bl <process> [process...] - list the breakpoints for each given process
        """
    debug = self.debug
    if arg == '*':
        if self.cmdprefix:
            raise CmdError('prefix not supported')
        breakpoints = debug.get_debugee_pids()
    else:
        targets = self.input_process_list(self.split_tokens(arg))
        if self.cmdprefix:
            targets.insert(0, self.input_process(self.cmdprefix))
        if not targets:
            if self.lastEvent is None:
                raise CmdError('no current process is set')
            targets = [self.lastEvent.get_pid()]
    for pid in targets:
        bplist = debug.get_process_code_breakpoints(pid)
        printed_process_banner = False
        if bplist:
            if not printed_process_banner:
                print('Process %d:' % pid)
                printed_process_banner = True
            for bp in bplist:
                address = repr(bp)[1:-1].replace('remote address ', '')
                print('  %s' % address)
        dbplist = debug.get_process_deferred_code_breakpoints(pid)
        if dbplist:
            if not printed_process_banner:
                print('Process %d:' % pid)
                printed_process_banner = True
            for label, action, oneshot in dbplist:
                if oneshot:
                    address = '  Deferred unconditional one-shot code breakpoint at %s'
                else:
                    address = '  Deferred unconditional code breakpoint at %s'
                address = address % label
                print('  %s' % address)
        bplist = debug.get_process_page_breakpoints(pid)
        if bplist:
            if not printed_process_banner:
                print('Process %d:' % pid)
                printed_process_banner = True
            for bp in bplist:
                address = repr(bp)[1:-1].replace('remote address ', '')
                print('  %s' % address)
        for tid in debug.system.get_process(pid).iter_thread_ids():
            bplist = debug.get_thread_hardware_breakpoints(tid)
            if bplist:
                print('Thread %d:' % tid)
                for bp in bplist:
                    address = repr(bp)[1:-1].replace('remote address ', '')
                    print('  %s' % address)
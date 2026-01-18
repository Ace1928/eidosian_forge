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
def do_register(self, arg):
    """
        [~thread] r - print(the value of all registers
        [~thread] r <register> - print(the value of a register
        [~thread] r <register>=<value> - change the value of a register
        [~thread] register - print(the value of all registers
        [~thread] register <register> - print(the value of a register
        [~thread] register <register>=<value> - change the value of a register
        """
    arg = arg.strip()
    if not arg:
        self.print_current_location()
    else:
        equ = arg.find('=')
        if equ >= 0:
            register = arg[:equ].strip()
            value = arg[equ + 1:].strip()
            if not value:
                value = '0'
            self.change_register(register, value)
        else:
            value = self.input_register(arg)
            if value is None:
                raise CmdError('unknown register: %s' % arg)
            try:
                label = None
                thread = self.get_thread_from_prefix()
                process = thread.get_process()
                module = process.get_module_at_address(value)
                if module:
                    label = module.get_label_at_address(value)
            except RuntimeError:
                label = None
            reg = arg.upper()
            val = HexDump.address(value)
            if label:
                print('%s: %s (%s)' % (reg, val, label))
            else:
                print('%s: %s' % (reg, val))
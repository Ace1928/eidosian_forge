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
def input_thread(self, token):
    targets = self.input_thread_list([token])
    if len(targets) == 0:
        raise CmdError('missing thread name or ID')
    if len(targets) > 1:
        msg = 'more than one thread with that name:\n'
        for tid in targets:
            msg += '\t%d\n' % tid
        msg = msg[:-len('\n')]
        raise CmdError(msg)
    return targets[0]
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
def print_thread_start(self, event):
    tid = event.get_tid()
    start = event.get_start_address()
    if start:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            start = event.get_process().get_label_at_address(start)
        print('Started thread %d at %s' % (tid, start))
    else:
        print('Attached to thread %d' % tid)
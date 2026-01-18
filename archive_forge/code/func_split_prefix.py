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
def split_prefix(self, line):
    prefix = None
    if line.startswith('~'):
        pos = line.find(' ')
        if pos == 1:
            pos = line.find(' ', pos + 1)
        if not pos < 0:
            prefix = line[1:pos].strip()
            line = line[pos:].strip()
    return (prefix, line)
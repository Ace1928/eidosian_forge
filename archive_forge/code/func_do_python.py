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
def do_python(self, arg):
    """
        # - spawn a python interpreter
        python - spawn a python interpreter
        # <statement> - execute a single python statement
        python <statement> - execute a single python statement
        """
    if self.cmdprefix:
        raise CmdError('prefix not allowed')
    if arg:
        try:
            compat.exec_(arg, globals(), locals())
        except Exception:
            traceback.print_exc()
    else:
        try:
            self._spawn_python_shell(arg)
        except Exception:
            e = sys.exc_info()[1]
            raise CmdError('unhandled exception when running Python console: %s' % e)
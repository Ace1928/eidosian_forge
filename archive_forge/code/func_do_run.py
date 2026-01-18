import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def do_run(self, arg):
    """run [args...]
        Restart the debugged python program. If a string is supplied
        it is split with "shlex", and the result is used as the new
        sys.argv.  History, breakpoints, actions and debugger options
        are preserved.  "restart" is an alias for "run".
        """
    if arg:
        import shlex
        argv0 = sys.argv[0:1]
        try:
            sys.argv = shlex.split(arg)
        except ValueError as e:
            self.error('Cannot run %s: %s' % (arg, e))
            return
        sys.argv[:0] = argv0
    raise Restart
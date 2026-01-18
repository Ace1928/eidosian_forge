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
def do_disable(self, arg):
    """disable bpnumber [bpnumber ...]
        Disables the breakpoints given as a space separated list of
        breakpoint numbers.  Disabling a breakpoint means it cannot
        cause the program to stop execution, but unlike clearing a
        breakpoint, it remains in the list of breakpoints and can be
        (re-)enabled.
        """
    args = arg.split()
    for i in args:
        try:
            bp = self.get_bpbynumber(i)
        except ValueError as err:
            self.error(err)
        else:
            bp.disable()
            self.message('Disabled %s' % bp)
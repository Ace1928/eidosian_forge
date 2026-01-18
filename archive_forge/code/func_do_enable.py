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
def do_enable(self, arg):
    """enable bpnumber [bpnumber ...]
        Enables the breakpoints given as a space separated list of
        breakpoint numbers.
        """
    args = arg.split()
    for i in args:
        try:
            bp = self.get_bpbynumber(i)
        except ValueError as err:
            self.error(err)
        else:
            bp.enable()
            self.message('Enabled %s' % bp)
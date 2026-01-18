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
def do_display(self, arg):
    """display [expression]

        Display the value of the expression if it changed, each time execution
        stops in the current frame.

        Without expression, list all display expressions for the current frame.
        """
    if not arg:
        self.message('Currently displaying:')
        for key, val in self.displaying.get(self.curframe, {}).items():
            self.message('%s: %s' % (key, self._safe_repr(val, key)))
    else:
        val = self._getval_except(arg)
        self.displaying.setdefault(self.curframe, {})[arg] = val
        self.message('display %s: %s' % (arg, self._safe_repr(val, arg)))
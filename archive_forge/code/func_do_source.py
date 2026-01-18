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
def do_source(self, arg):
    """source expression
        Try to get source code for the given object and display it.
        """
    try:
        obj = self._getval(arg)
    except:
        return
    try:
        lines, lineno = self._getsourcelines(obj)
    except (OSError, TypeError) as err:
        self.error(err)
        return
    self._print_lines(lines, lineno)
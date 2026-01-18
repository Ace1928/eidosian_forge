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
def do_whatis(self, arg):
    """whatis arg
        Print the type of the argument.
        """
    try:
        value = self._getval(arg)
    except:
        return
    code = None
    try:
        code = value.__func__.__code__
    except Exception:
        pass
    if code:
        self.message('Method %s' % code.co_name)
        return
    try:
        code = value.__code__
    except Exception:
        pass
    if code:
        self.message('Function %s' % code.co_name)
        return
    if value.__class__ is type:
        self.message('Class %s.%s' % (value.__module__, value.__qualname__))
        return
    self.message(type(value))
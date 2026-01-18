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
def do_retval(self, arg):
    """retval
        Print the return value for the last return of a function.
        """
    if '__return__' in self.curframe_locals:
        self.message(self._safe_repr(self.curframe_locals['__return__'], 'retval'))
    else:
        self.error('Not yet returned!')
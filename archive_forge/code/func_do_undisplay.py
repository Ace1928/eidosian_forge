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
def do_undisplay(self, arg):
    """undisplay [expression]

        Do not display the expression any more in the current frame.

        Without expression, clear all display expressions for the current frame.
        """
    if arg:
        try:
            del self.displaying.get(self.curframe, {})[arg]
        except KeyError:
            self.error('not displaying %s' % arg)
    else:
        self.displaying.pop(self.curframe, None)
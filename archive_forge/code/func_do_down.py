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
def do_down(self, arg):
    """d(own) [count]
        Move the current frame count (default one) levels down in the
        stack trace (to a newer frame).
        """
    if self.curindex + 1 == len(self.stack):
        self.error('Newest frame')
        return
    try:
        count = int(arg or 1)
    except ValueError:
        self.error('Invalid frame count (%s)' % arg)
        return
    if count < 0:
        newframe = len(self.stack) - 1
    else:
        newframe = min(len(self.stack) - 1, self.curindex + count)
    self._select_frame(newframe)
import io
import linecache
import os
import sys
import sysconfig
import token
import tokenize
import inspect
import gc
import dis
import pickle
from time import monotonic as _time
import threading
def globaltrace_countfuncs(self, frame, why, arg):
    """Handler for call events.

        Adds (filename, modulename, funcname) to the self._calledfuncs dict.
        """
    if why == 'call':
        this_func = self.file_module_function_of(frame)
        self._calledfuncs[this_func] = 1
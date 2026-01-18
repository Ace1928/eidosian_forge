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
def globaltrace_lt(self, frame, why, arg):
    """Handler for call events.

        If the code block being entered is to be ignored, returns `None',
        else returns self.localtrace.
        """
    if why == 'call':
        code = frame.f_code
        filename = frame.f_globals.get('__file__', None)
        if filename:
            modulename = _modname(filename)
            if modulename is not None:
                ignore_it = self.ignore.names(filename, modulename)
                if not ignore_it:
                    if self.trace:
                        print(' --- modulename: %s, funcname: %s' % (modulename, code.co_name))
                    return self.localtrace
        else:
            return None
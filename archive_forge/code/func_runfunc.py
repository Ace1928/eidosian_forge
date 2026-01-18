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
def runfunc(self, func, /, *args, **kw):
    result = None
    if not self.donothing:
        sys.settrace(self.globaltrace)
    try:
        result = func(*args, **kw)
    finally:
        if not self.donothing:
            sys.settrace(None)
    return result
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
def localtrace_count(self, frame, why, arg):
    if why == 'line':
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        key = (filename, lineno)
        self.counts[key] = self.counts.get(key, 0) + 1
    return self.localtrace
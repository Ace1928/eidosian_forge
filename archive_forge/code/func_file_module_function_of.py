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
def file_module_function_of(self, frame):
    code = frame.f_code
    filename = code.co_filename
    if filename:
        modulename = _modname(filename)
    else:
        modulename = None
    funcname = code.co_name
    clsname = None
    if code in self._caller_cache:
        if self._caller_cache[code] is not None:
            clsname = self._caller_cache[code]
    else:
        self._caller_cache[code] = None
        funcs = [f for f in gc.get_referrers(code) if inspect.isfunction(f)]
        if len(funcs) == 1:
            dicts = [d for d in gc.get_referrers(funcs[0]) if isinstance(d, dict)]
            if len(dicts) == 1:
                classes = [c for c in gc.get_referrers(dicts[0]) if hasattr(c, '__bases__')]
                if len(classes) == 1:
                    clsname = classes[0].__name__
                    self._caller_cache[code] = clsname
    if clsname is not None:
        funcname = '%s.%s' % (clsname, funcname)
    return (filename, modulename, funcname)
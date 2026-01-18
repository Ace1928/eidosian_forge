from collections import namedtuple
from functools import singledispatch as simplegeneric
import importlib
import importlib.util
import importlib.machinery
import os
import os.path
import sys
from types import ModuleType
import warnings
def iter_modules(self, prefix=''):
    if self.path is None or not os.path.isdir(self.path):
        return
    yielded = {}
    import inspect
    try:
        filenames = os.listdir(self.path)
    except OSError:
        filenames = []
    filenames.sort()
    for fn in filenames:
        modname = inspect.getmodulename(fn)
        if modname == '__init__' or modname in yielded:
            continue
        path = os.path.join(self.path, fn)
        ispkg = False
        if not modname and os.path.isdir(path) and ('.' not in fn):
            modname = fn
            try:
                dircontents = os.listdir(path)
            except OSError:
                dircontents = []
            for fn in dircontents:
                subname = inspect.getmodulename(fn)
                if subname == '__init__':
                    ispkg = True
                    break
            else:
                continue
        if modname and '.' not in modname:
            yielded[modname] = 1
            yield (prefix + modname, ispkg)
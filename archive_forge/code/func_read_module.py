import sys, os
import textwrap
def read_module(self, modname, mode='careful'):
    __import__(modname)
    mod = sys.modules[modname]
    self._update(vars(mod), mode)
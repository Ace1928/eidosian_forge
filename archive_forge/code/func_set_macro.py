import sys, os
from distutils.errors import \
from distutils.ccompiler import \
from distutils import log
def set_macro(self, macro, path, key):
    for base in HKEYS:
        d = read_values(base, path)
        if d:
            self.macros['$(%s)' % macro] = d[key]
            break
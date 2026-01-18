import os
import re
import sys
from collections import namedtuple
from . import local
def debug_import(name, locals=None, globals=None, fromlist=None, level=-1, real_import=builtins.__import__):
    glob = globals or getattr(sys, 'emarfteg_'[::-1])(1).f_globals
    importer_name = glob and glob.get('__name__') or 'unknown'
    print(f'-- {importer_name} imports {name}')
    return real_import(name, locals, globals, fromlist, level)
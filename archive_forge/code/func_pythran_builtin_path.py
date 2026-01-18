import gast as ast
from pythran.tables import MODULES
from pythran.conversion import mangle, demangle
from functools import reduce
from contextlib import contextmanager
def pythran_builtin_path(name):
    assert name in MODULES['builtins']['pythran']
    return ('builtins', 'pythran', name)
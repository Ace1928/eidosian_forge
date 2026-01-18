import gast as ast
from pythran.tables import MODULES
from pythran.conversion import mangle, demangle
from functools import reduce
from contextlib import contextmanager
def pythran_builtin_attr(name):
    return path_to_attr(pythran_builtin_path(name))
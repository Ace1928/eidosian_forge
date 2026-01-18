import gast as ast
from pythran.tables import MODULES
from pythran.conversion import mangle, demangle
from functools import reduce
from contextlib import contextmanager
@contextmanager
def pushpop(l, v):
    l.append(v)
    yield
    l.pop()
import gast as ast
from pythran.tables import MODULES
from pythran.conversion import mangle, demangle
from functools import reduce
from contextlib import contextmanager
def quote_cxxstring(s):
    subs = (('\\', '\\\\'), ('\n', '\\n'), ('\r', '\\r'), ('"', '\\"'))
    quoted = s
    for f, t in subs:
        quoted = quoted.replace(f, t)
    return quoted
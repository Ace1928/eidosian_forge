import gast as ast
from pythran.tables import MODULES
from pythran.conversion import mangle, demangle
from functools import reduce
from contextlib import contextmanager
def isextslice(node):
    if not isinstance(node, ast.Tuple):
        return False
    return any((isinstance(elt, ast.Slice) for elt in node.elts))
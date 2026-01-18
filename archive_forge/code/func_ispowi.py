import gast as ast
from pythran.tables import MODULES
from pythran.conversion import mangle, demangle
from functools import reduce
from contextlib import contextmanager
def ispowi(node):
    if not isinstance(node.op, ast.Pow):
        return False
    attr = 'right' if isinstance(node, ast.BinOp) else 'value'
    if not isintegral(getattr(node, attr)):
        return False
    return getattr(node, attr).value >= 0
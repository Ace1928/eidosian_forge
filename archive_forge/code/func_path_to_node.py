import gast as ast
from pythran.tables import MODULES
from pythran.conversion import mangle, demangle
from functools import reduce
from contextlib import contextmanager
def path_to_node(path):
    """
    Retrieve a symbol in MODULES based on its path
    >>> path = ('math', 'pi')
    >>> path_to_node(path) #doctest: +ELLIPSIS
    <pythran.intrinsic.ConstantIntr object at 0x...>
    """
    if len(path) == 1:
        return MODULES[path[0]]
    else:
        return path_to_node(path[:-1])[path[-1]]
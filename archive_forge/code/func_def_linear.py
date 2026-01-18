from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
def def_linear(fun):
    """Flags that a function is linear wrt all args"""
    defjvp_argnum(fun, lambda argnum, g, ans, args, kwargs: fun(*subval(args, argnum, g), **kwargs))
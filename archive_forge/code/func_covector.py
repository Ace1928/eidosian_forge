from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
@primitive
def covector(self, x):
    return self._covector(x)
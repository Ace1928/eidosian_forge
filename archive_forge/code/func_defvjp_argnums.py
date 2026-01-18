from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
def defvjp_argnums(fun, vjpmaker):
    primitive_vjps[fun] = vjpmaker
from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
def staged_vjpmaker(ans, *args, **kwargs):

    def vjp(g):
        vs, gvs = (vspace(args[argnum]), vspace(g))
        return vjpmaker(g, ans, vs, gvs, *args, **kwargs)
    return vjp
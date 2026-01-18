from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
def defjvp_argnum(fun, jvpmaker):

    def jvp_argnums(argnums, gs, ans, args, kwargs):
        return sum_outgrads((jvpmaker(argnum, g, ans, args, kwargs) for argnum, g in zip(argnums, gs)))
    defjvp_argnums(fun, jvp_argnums)
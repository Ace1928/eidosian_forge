from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
def defvjp_unstaged(vjpmaker, argnum=0):
    warnings.warn(deprecation_msg)

    def staged_vjpmaker(ans, *args, **kwargs):

        def vjp(g):
            vs, gvs = (vspace(args[argnum]), vspace(g))
            return vjpmaker(g, ans, vs, gvs, *args, **kwargs)
        return vjp
    vjpfuns[argnum] = staged_vjpmaker
    argnums, vjpmakers = zip(*[(argnum, vjpfuns[argnum]) for argnum in sorted(vjpfuns.keys())])
    defvjp(primitive_fun, *vjpmakers, argnums=argnums)
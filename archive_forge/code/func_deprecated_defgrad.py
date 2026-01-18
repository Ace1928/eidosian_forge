from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
def deprecated_defgrad(primitive_fun):
    deprecation_msg = deprecated_defvjp_message.format('defgrad')
    gradfuns = {}

    def defgrad(gradfun, argnum=0):
        warnings.warn(deprecation_msg)
        gradfuns[argnum] = gradfun
        argnums, vjpmakers = zip(*[(argnum, gradfuns[argnum]) for argnum in sorted(gradfuns.keys())])
        defvjp(primitive_fun, *vjpmakers, argnums=argnums)
    return defgrad
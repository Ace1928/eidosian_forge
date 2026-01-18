from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
def primitive_with_deprecation_warnings(f_raw):
    f_wrapped = primitive_(f_raw)
    f_wrapped.defvjp = deprecated_defvjp(f_wrapped)
    f_wrapped.defvjp_is_zero = deprecated_defvjp_is_zero(f_wrapped)
    f_wrapped.defgrad = deprecated_defgrad(f_wrapped)
    return f_wrapped
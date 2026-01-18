from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
def sum_outgrads(gs):
    return reduce(add_outgrads, gs, None)[0]
import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core import Suffix, Var, Constraint, Piecewise, Block
from pyomo.core import Expression, Param
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.block import IndexedBlock, SortComponents
from pyomo.dae import ContinuousSet, DAE_Error
from pyomo.common.formatting import tostr
from io import StringIO
def create_partial_expression(scheme, expr, ind, loc):
    """
    This method returns a function which applies a discretization scheme
    to an expression along a particular indexing set. This is admittedly a
    convoluted looking implementation. The idea is that we only apply a
    discretization scheme to one indexing set at a time but we also want
    the function to be expanded over any other indexing sets.
    """

    def _fun(*args):
        return scheme(lambda i: expr(*args[0:loc] + (i,) + args[loc + 1:]), ind)
    return lambda *args: _fun(*args)(args[loc])
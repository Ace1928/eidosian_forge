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
def get_index_information(var, ds):
    """
    This method will find the index location of the set ds in the var,
    return a list of the non_ds indices and return a function that can be
    used to access specific indices in var indexed by a ContinuousSet by
    specifying the finite element and collocation point. Users of this
    method should have already confirmed that ds is an indexing set of var
    and that it's a ContinuousSet
    """
    indargs = []
    dsindex = 0
    if var.dim() != 1:
        indCount = 0
        for index in var.index_set().subsets():
            if isinstance(index, ContinuousSet):
                if index is ds:
                    dsindex = indCount
                else:
                    indargs.append(index)
                indCount += 1
            else:
                indargs.append(index)
                indCount += index.dimen
    if indargs == []:
        non_ds = (None,)
    elif len(indargs) > 1:
        non_ds = tuple((a for a in indargs))
    else:
        non_ds = (indargs[0],)
    if None in non_ds:
        tmpidx = (None,)
    elif len(non_ds) == 1:
        tmpidx = non_ds[0]
    else:
        tmpidx = non_ds[0].cross(*non_ds[1:])
    idx = lambda n, i, k: _get_idx(dsindex, ds, n, i, k)
    info = dict()
    info['non_ds'] = tmpidx
    info['index function'] = idx
    return info
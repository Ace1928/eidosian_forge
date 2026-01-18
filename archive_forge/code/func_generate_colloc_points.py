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
def generate_colloc_points(ds, tau):
    """
    This function adds collocation points between the finite elements
    in the differential set
    """
    fes = list(ds)
    for i in range(1, len(fes)):
        h = fes[i] - fes[i - 1]
        for j in range(len(tau)):
            if tau[j] == 1 or tau[j] == 0:
                continue
            pt = fes[i - 1] + h * tau[j]
            pt = round(pt, 6)
            if pt not in ds:
                ds.add(pt)
                ds.set_changed(True)
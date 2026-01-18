from pyomo.gdp import GDP_Error, Disjunction
from pyomo.gdp.disjunct import _DisjunctData, Disjunct
import pyomo.core.expr as EXPR
from pyomo.core.base.component import _ComponentBase
from pyomo.core import (
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentMap, ComponentSet, OrderedSet
from pyomo.opt import TerminationCondition, SolverStatus
from weakref import ref as weakref_ref
from collections import defaultdict
import logging
def parent_disjunct(self, u):
    """Returns the parent Disjunct of u, or None if u is the
        closest-to-root Disjunct in the forest.

        Arg:
            u : A node in the forest
        """
    if u.ctype is Disjunct:
        return self.parent(self.parent(u))
    else:
        return self.parent(u)
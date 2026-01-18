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
def root_disjunct(self, u):
    """Returns the highest parent Disjunct in the hierarchy, or None if
        the component is not nested.

        Arg:
            u : A node in the tree
        """
    rootmost_disjunct = None
    parent = self.parent(u)
    while True:
        if parent is None:
            return rootmost_disjunct
        if parent.ctype is Disjunct:
            rootmost_disjunct = parent
        parent = self.parent(parent)
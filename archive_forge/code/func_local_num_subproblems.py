from pyomo.core.base.block import _BlockData, declare_custom_block
import pyomo.environ as pyo
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet
import logging
def local_num_subproblems(self):
    return len(self.subproblems)
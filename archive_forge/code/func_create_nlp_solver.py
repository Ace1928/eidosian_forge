from collections import namedtuple
from pyomo.core.base.objective import Objective
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.opt import SolverResults, TerminationCondition
from pyomo.common.dependencies import (
def create_nlp_solver(self, **kwds):
    nlp = self.get_nlp()
    solver = SecantNewtonNlpSolver(nlp, **kwds)
    return solver
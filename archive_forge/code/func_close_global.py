import logging
import re
import sys
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import ApplicationError
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.numvalue import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.opt.base import SolverFactory
from pyomo.core.base.suffix import Suffix
import pyomo.core.base.var
def close_global(self):
    """Frees all Gurobi models used by this solver, and frees the global
        default Gurobi environment.

        The default environment is used by all ``GurobiDirect`` solvers started
        with ``manage_env=False`` (the default). To guarantee that all Gurobi
        resources are freed, all instantiated ``GurobiDirect`` solvers must also
        be correctly closed.

        The following example will free all Gurobi resources assuming the user did
        not create any other models (e.g. via another ``GurobiDirect`` object with
        ``manage_env=False``)::

            opt = SolverFactory('gurobi', solver_io='python')
            try:
                opt.solve(model)
            finally:
                opt.close_global()
            # All Gurobi models created by `opt` are freed and the default
            # Gurobi environment is closed
        """
    self.close()
    with capture_output(capture_fd=True):
        gurobipy.disposeDefaultEnv()
    GurobiDirect._default_env_started = False
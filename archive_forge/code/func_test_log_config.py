import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
import math
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import (
from pyomo.contrib.pyros.master_problem_methods import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
from pyomo.environ import (
import logging
from itertools import chain
def test_log_config(self):
    """
        Test method for logging PyROS solver config dict.
        """
    pyros_solver = SolverFactory('pyros')
    config = pyros_solver.CONFIG(dict(nominal_uncertain_param_vals=[0.5]))
    with LoggingIntercept(level=logging.INFO) as LOG:
        pyros_solver._log_config(logger=logger, config=config, level=logging.INFO)
    ans = 'Solver options:\n time_limit=None\n keepfiles=False\n tee=False\n load_solution=True\n objective_focus=<ObjectiveType.nominal: 2>\n nominal_uncertain_param_vals=[0.5]\n decision_rule_order=0\n solve_master_globally=False\n max_iter=-1\n robust_feasibility_tolerance=0.0001\n separation_priority_order={}\n progress_logger=<PreformattedLogger pyomo.contrib.pyros (INFO)>\n backup_local_solvers=[]\n backup_global_solvers=[]\n subproblem_file_directory=None\n bypass_local_separation=False\n bypass_global_separation=False\n p_robustness={}\n' + '-' * 78 + '\n'
    logged_str = LOG.getvalue()
    self.assertEqual(logged_str, ans, msg='Logger output for PyROS solver config (default case) does not match expected result.')
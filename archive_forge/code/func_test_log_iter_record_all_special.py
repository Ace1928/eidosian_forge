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
def test_log_iter_record_all_special(self):
    """
        Test iteration log record in event DR polishing and global
        separation failed.
        """
    iter_record = IterationLogRecord(iteration=4, objective=1.234567, first_stage_var_shift=2.3456789e-08, second_stage_var_shift=3.456789e-07, dr_var_shift=1.234567e-07, num_violated_cons=10, max_violation=0.007654321, elapsed_time=21.2, dr_polishing_success=False, all_sep_problems_solved=False, global_separation=True)
    ans = '4     1.2346e+00  2.3457e-08   3.4568e-07*  10+     7.6543e-03g  21.200       \n'
    with LoggingIntercept(level=logging.INFO) as LOG:
        iter_record.log(logger.info)
    result = LOG.getvalue()
    self.assertEqual(ans, result, msg='Iteration log record message does not match expected result')
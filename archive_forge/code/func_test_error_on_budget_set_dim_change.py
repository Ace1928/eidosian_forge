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
def test_error_on_budget_set_dim_change(self):
    """
        BudgetSet dimension is considered immutable.
        Test ValueError raised when attempting to alter the
        budget set dimension.
        """
    budget_mat = [[1, 0, 1], [0, 1, 0]]
    budget_rhs_vec = [1, 3]
    bu_set = BudgetSet(budget_mat, budget_rhs_vec)
    exc_str = '.*must have 3 columns to match set dimension \\(provided.*1 columns\\)'
    with self.assertRaisesRegex(ValueError, exc_str):
        bu_set.budget_membership_mat = [[1], [1]]
    exc_str = '.*must have 3 entries to match set dimension \\(provided.*4 entries\\)'
    with self.assertRaisesRegex(ValueError, exc_str):
        bu_set.origin = [1, 2, 1, 0]
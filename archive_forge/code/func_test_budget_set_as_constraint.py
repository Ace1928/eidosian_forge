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
def test_budget_set_as_constraint(self):
    """
        The set_as_constraint method must return an indexed uncertainty_set_constr
        which has as many elements at their are dimensions in A.
        """
    m = ConcreteModel()
    m.p1 = Var(initialize=1)
    m.p2 = Var(initialize=1)
    m.uncertain_params = [m.p1, m.p2]
    budget_membership_mat = [[1 for i in range(len(m.uncertain_params))]]
    rhs_vec = [0.1 * len(m.uncertain_params) + sum((p.value for p in m.uncertain_params))]
    budget_set = BudgetSet(budget_membership_mat=budget_membership_mat, rhs_vec=rhs_vec)
    m.uncertainty_set_constr = budget_set.set_as_constraint(uncertain_params=m.uncertain_params)
    self.assertEqual(len(budget_set.coefficients_mat), len(m.uncertainty_set_constr.index_set()), msg="Number of budget set constraints should be equal to the number of rows in the 'coefficients_mat' attribute")
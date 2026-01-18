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
def test_identify_objectives_var_expr(self):
    """
        Test first and second-stage objective identification
        for an objective expression consisting only of a Var.
        """
    m = ConcreteModel()
    m.p = Param(range(4), initialize=1, mutable=True)
    m.q = Param(initialize=1)
    m.x = Var(range(4))
    m.obj = Objective(expr=m.x[1])
    m.util = Block()
    m.util.first_stage_variables = list(m.x.values())
    m.util.second_stage_variables = list()
    m.util.uncertain_params = list()
    identify_objective_functions(m, m.obj)
    fsv_in_second_stg_obj = list((v.name for v in identify_variables(m.second_stage_objective)))
    self.assertTrue(list(identify_variables(m.first_stage_objective)) == [m.x[1]])
    self.assertFalse(fsv_in_second_stg_obj, f'Second stage objective contains variable(s) {fsv_in_second_stg_obj}')
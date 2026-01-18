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
@unittest.skipIf(not scipy_available, 'Scipy is not available.')
def test_correct_num_dr_vars_quadratic(self):
    """
        Test DR variable setup routines declare the correct
        number of DR coefficient variables, quadratic DR case.
        """
    model_data = ROSolveResults()
    model_data.working_model = m = self.make_simple_test_model()
    config = Bunch()
    config.decision_rule_order = 2
    add_decision_rule_variables(model_data=model_data, config=config)
    num_params = len(m.util.uncertain_params)
    correct_num_dr_vars = 1 + num_params + sp.special.comb(num_params, 2, repetition=True, exact=True)
    for indexed_dr_var in m.util.decision_rule_vars:
        self.assertEqual(len(indexed_dr_var), correct_num_dr_vars, msg=f'Number of decision rule coefficient variables in indexed Var object {indexed_dr_var.name!r}does not match correct value.')
    self.assertEqual(len(ComponentSet(m.util.decision_rule_vars)), len(m.util.second_stage_variables), msg='Number of unique indexed DR variable components should equal number of second-stage variables.')
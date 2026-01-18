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
@unittest.skipUnless(SolverFactory('cbc').available(exception_flag=False), 'LP solver CBC not available')
def test_factor_model_parameter_bounds_correct(self):
    """
        If LP solver is available, test parameter bounds method
        for factor model set is correct (check against
        results from an LP solver).
        """
    solver = SolverFactory('cbc')
    fset1 = FactorModelSet(origin=[0, 0], number_of_factors=3, psi_mat=[[1, -1, 1], [1, 0.1, 1]], beta=1 / 6)
    fset2 = FactorModelSet(origin=[0], number_of_factors=3, psi_mat=[[1, 6, 8]], beta=1 / 2)
    fset3 = FactorModelSet(origin=[1], number_of_factors=2, psi_mat=[[1, 2]], beta=1 / 4)
    fset4 = FactorModelSet(origin=[1], number_of_factors=3, psi_mat=[[-1, -6, -8]], beta=1 / 2)
    for fset in [fset1, fset2, fset3, fset4]:
        param_bounds = fset.parameter_bounds
        lp_param_bounds = eval_parameter_bounds(fset, solver)
        self.assertTrue(np.allclose(param_bounds, lp_param_bounds), msg=f'Parameter bounds not consistent with LP values for FactorModelSet with parameterization:\nF={fset.number_of_factors},\nbeta={fset.beta},\npsi_mat={fset.psi_mat},\norigin={fset.origin}.')
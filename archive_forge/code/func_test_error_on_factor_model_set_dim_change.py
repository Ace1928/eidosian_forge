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
def test_error_on_factor_model_set_dim_change(self):
    """
        Test ValueError raised when attempting to change FactorModelSet
        dimension (by changing number of entries in origin
        or number of rows of psi_mat).
        """
    origin = [0, 0, 0]
    number_of_factors = 2
    psi_mat = [[1, 0], [0, 1], [1, 1]]
    beta = 0.5
    fset = FactorModelSet(origin, number_of_factors, psi_mat, beta)
    exc_str = 'should be of shape \\(3, 2\\) to match.*dimensions \\(provided shape \\(2, 2\\)\\)'
    with self.assertRaisesRegex(ValueError, exc_str):
        fset.psi_mat = [[1, 0], [1, 2]]
    exc_str = 'Attempting.*factor model set of dimension 3 to value of dimension 2'
    with self.assertRaisesRegex(ValueError, exc_str):
        fset.origin = [1, 3]
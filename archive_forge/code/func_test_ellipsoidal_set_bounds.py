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
def test_ellipsoidal_set_bounds(self):
    """Check `EllipsoidalSet` parameter bounds method correct."""
    cov = [[2, 1], [1, 2]]
    scales = [0.5, 2]
    mean = [1, 1]
    for scale in scales:
        ell = EllipsoidalSet(center=mean, shape_matrix=cov, scale=scale)
        bounds = ell.parameter_bounds
        actual_bounds = list()
        for idx, val in enumerate(mean):
            diff = (cov[idx][idx] * scale) ** 0.5
            actual_bounds.append((val - diff, val + diff))
        self.assertTrue(np.allclose(np.array(bounds), np.array(actual_bounds)), msg=f'EllipsoidalSet bounds {bounds} do not match their actual values {actual_bounds} (for scale {scale} and shape matrix {cov}). Check the `parameter_bounds` method for the EllipsoidalSet.')
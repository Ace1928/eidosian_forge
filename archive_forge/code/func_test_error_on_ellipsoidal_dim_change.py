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
def test_error_on_ellipsoidal_dim_change(self):
    """
        EllipsoidalSet dimension is considered immutable.
        Test ValueError raised when center size is not equal
        to set dimension.
        """
    invalid_center = [0, 0]
    shape_matrix = [[1, 0], [0, 1]]
    scale = 2
    eset = EllipsoidalSet([0, 0], shape_matrix, scale)
    exc_str = 'Attempting to set.*dimension 2 to value of dimension 3'
    with self.assertRaisesRegex(ValueError, exc_str):
        eset.center = [0, 0, 0]
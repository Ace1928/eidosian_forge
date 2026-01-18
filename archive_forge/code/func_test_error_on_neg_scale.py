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
def test_error_on_neg_scale(self):
    """
        Test ValueError raised if scale attribute set to negative
        value.
        """
    center = [0, 0]
    shape_matrix = [[1, 0], [0, 2]]
    neg_scale = -1
    exc_str = '.*must be a non-negative real \\(provided.*-1\\)'
    with self.assertRaisesRegex(ValueError, exc_str):
        EllipsoidalSet(center, shape_matrix, neg_scale)
    eset = EllipsoidalSet(center, shape_matrix, scale=2)
    with self.assertRaisesRegex(ValueError, exc_str):
        eset.scale = neg_scale
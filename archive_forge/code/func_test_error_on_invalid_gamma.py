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
def test_error_on_invalid_gamma(self):
    """
        Cardinality set gamma attribute should be a float-like
        between 0 and the set dimension.

        Check ValueError raised if gamma attribute is set
        to an invalid value.
        """
    origin = [0, 0]
    positive_deviation = [1, 1]
    gamma = 3
    exc_str = ".*attribute 'gamma' must be a real number between 0 and dimension 2 \\(provided value 3\\)"
    with self.assertRaisesRegex(ValueError, exc_str):
        CardinalitySet(origin, positive_deviation, gamma)
    cset = CardinalitySet(origin, positive_deviation, gamma=2)
    with self.assertRaisesRegex(ValueError, exc_str):
        cset.gamma = gamma
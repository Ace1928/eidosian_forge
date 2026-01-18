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
def test_error_on_empty_last_dimension(self):
    """
        Check ValueError raised when last dimension of BoxSet bounds is
        empty.
        """
    empty_2d_arr = [[], [], []]
    exc_str = 'Last dimension of argument `bounds` must be non-empty \\(detected shape \\(3, 0\\)\\)'
    with self.assertRaisesRegex(ValueError, exc_str):
        BoxSet(bounds=empty_2d_arr)
    bset = BoxSet([[1, 2], [3, 4]])
    with self.assertRaisesRegex(ValueError, exc_str):
        bset.bounds = empty_2d_arr
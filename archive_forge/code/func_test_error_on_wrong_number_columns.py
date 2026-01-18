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
def test_error_on_wrong_number_columns(self):
    """
        BoxSet bounds should be a 2D array-like with 2 columns.
        ValueError raised if number columns wrong
        """
    three_col_arr = [[1, 2, 3], [4, 5, 6]]
    exc_str = "Attribute 'bounds' should be of shape \\(\\.{3},2\\), but detected shape \\(\\.{3},3\\)"
    with self.assertRaisesRegex(ValueError, exc_str):
        BoxSet(three_col_arr)
    bset = BoxSet([[1, 2], [3, 4]])
    with self.assertRaisesRegex(ValueError, exc_str):
        bset.bounds = three_col_arr
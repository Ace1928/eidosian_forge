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
def test_error_on_bounds_with_nan_or_inf(self):
    """
        Box set bounds set to array-like with inf or nan.
        """
    bset = BoxSet(bounds=[[1, 2], [3, 4]])
    for val_str in ['inf', 'nan']:
        bad_bounds = [[1, float(val_str)], [2, 3]]
        exc_str = f"Entry '{val_str}' of the argument `bounds` is not a finite numeric value"
        with self.assertRaisesRegex(ValueError, exc_str):
            BoxSet(bad_bounds)
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = bad_bounds
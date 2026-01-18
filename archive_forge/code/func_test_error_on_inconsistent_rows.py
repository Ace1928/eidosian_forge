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
def test_error_on_inconsistent_rows(self):
    """
        Number of rows of budget membership mat is immutable.
        Similarly, size of rhs_vec is immutable.
        Check ValueError raised in event of attempted change.
        """
    coeffs_mat_exc_str = ".*must have 2 rows to match shape of attribute 'rhs_vec' \\(provided.*3 rows\\)"
    rhs_vec_exc_str = ".*must have 2 entries to match shape of attribute 'coefficients_mat' \\(provided.*3 entries\\)"
    with self.assertRaisesRegex(ValueError, rhs_vec_exc_str):
        PolyhedralSet([[1, 2], [3, 4]], rhs_vec=[1, 3, 3])
    pset = PolyhedralSet([[1, 2], [3, 4]], rhs_vec=[1, 3])
    with self.assertRaisesRegex(ValueError, coeffs_mat_exc_str):
        pset.coefficients_mat = [[1, 2], [3, 4], [5, 6]]
    with self.assertRaisesRegex(ValueError, rhs_vec_exc_str):
        pset.rhs_vec = [1, 3, 2]
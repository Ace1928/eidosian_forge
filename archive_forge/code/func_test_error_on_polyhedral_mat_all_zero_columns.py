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
def test_error_on_polyhedral_mat_all_zero_columns(self):
    """
        Test ValueError raised if budget membership mat
        has a column with all zeros.
        """
    invalid_col_mat = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
    rhs_vec = [1, 1, 2]
    exc_str = '.*all entries zero in columns at indexes: 0, 1.*'
    with self.assertRaisesRegex(ValueError, exc_str):
        PolyhedralSet(invalid_col_mat, rhs_vec)
    pset = PolyhedralSet([[1, 0, 1], [1, 1, 0], [1, 1, 1]], rhs_vec)
    with self.assertRaisesRegex(ValueError, exc_str):
        pset.coefficients_mat = invalid_col_mat